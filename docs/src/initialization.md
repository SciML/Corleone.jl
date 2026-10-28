# Initializing shooting variables

Set `default` on a `Tunable` or `Constant` annotation to initialize its scheduled
values during `LuxCore.setup`. The same forms work for states and parameters:

| Default | Behavior |
|---|---|
| Scalar | Repeat the value for every entry. |
| `AbstractVector` | Use the supplied entries in schedule order. |
| `(rng, n::Int) -> values` | Generate `n` values using the setup RNG. |
| `(rng, times::AbstractVector) -> values` | Generate values using the actual initialization times. |

Callbacks must return an `AbstractVector` with the required number of numeric
values. Count callbacks take precedence if both signatures are applicable. In
particular, an untyped callback `(rng, n) -> ...` retains its count-based behavior;
annotate the second argument with `::AbstractVector` to request timepoints.
Callback exceptions propagate directly; a failed callback is not retried with a
different argument.

## Several variables on one grid

A vector selector supplies one default per selected variable. Scalar defaults
repeat over that variable's schedule; vector defaults specify its time series:

```julia
Tunable([2, 5, 11]; default = [0.2, 0.5, 1.1]) => switching_times
Tunable([2, 5]; default = [[0.2, 0.3], [0.5, 0.6]]) => [1.0]
```

Vectors of symbols work the same way. Annotations resolve into scalar definitions.
Variables with equal grids are grouped automatically, including definitions from
separate annotations. Each `InitialCondition` or `PiecewiseConstant` group has one
lookup and a NamedTuple of scalar `ParameterCollection` children. Fixed members
wrap their own collection in `FixedLayer`, so fixed and tunable variables can share
a lookup. Every buffer remains a vector of scalar values.

The outer collection uses the first member's symbol as the group key. For a group
containing parameters 2, 5 and 11, parameter 5's tunable buffer is
`ps.parameter.parameters.p₂.parameter.p₅.ps`. State groups use the equivalent
`initialcondition.initialconditions` path. Single-member groups retain their
previous layout. The outer Lux state stores a permutation for canonical output
assembly. Generated code assembles scalar entries without concatenating tracked
arrays.

Changing a group's grid changes the schedule of all its members. Reconstruct the
layer to move only some members to another grid. Parameter splitting expands fixed
buffers inside mixed groups as needed to preserve their values; continuity
constraints still include only tunable members.

## State reset values

Each variable has its own reset grid. There is one initial value per grid entry,
and the callback receives the grid in its supplied order. A variable without a
definition passes through its incoming value. An explicit empty or `nothing` state
grid initializes an empty value vector.

```jldoctest
julia> using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random

julia> prob = ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 3.0), [1.0]);

julia> resets = Tunable(1; kind = Val(:variable),
           default = (rng, times::AbstractVector) -> 10 .* times) => [1.0, 2.0];

julia> layer = ShootingLayer(prob, Tsit5(), resets);

julia> ps, st = LuxCore.setup(MersenneTwister(42), layer);

julia> ps.initialcondition.initialconditions.u₁.parameter.ps
2-element Vector{Float64}:
 10.0
 20.0
```

A time-based callback can evaluate a reference solution or an interpolant:

```julia
Tunable(:x; kind = Val(:variable),
    default = (rng, times::AbstractVector) -> [reference(t)[1] for t in times],
) => reset_times
```

The callback must supply the sampling operation. Corleone's `Trajectory` stores
actual saved samples and does not itself provide interpolation.

## Parameter interval values

For parameters, `n` switching times define `n + 1` intervals. A vector default must
contain one entry per interval. Time-based callbacks receive the problem start
followed by the supplied switching times:

```julia
# On a problem with tspan = (2.0, 6.0), the callback receives [2.0, 3.0, 5.0].
Tunable(1; kind = Val(:parameter),
    default = (rng, times::AbstractVector) -> times.^2,
) => [3.0, 5.0]
```

A parameter with a `nothing` switching grid has one value; its time-based callback
receives `[tspan[1]]`. The internal `PiecewiseConstant` constructor requires an
explicit `initialization_times` keyword for time-based callbacks because it has no
problem time span. The public `Parameter` and `ShootingLayer` constructors supply
these times automatically.

## Validation and ownership

Scalar and vector defaults are checked against annotation bounds when constructing
the layer. Callback results are evaluated once per initialized variable during
setup, then checked for vector shape, length, and bounds. This also applies to
`Constant` annotations, whose initialized values live in Lux state. Initialization
adds no work to subsequent forward evaluations.

Supplied vectors and callback results are retained by reference. To obtain
independent arrays on repeated setup calls, return a fresh array from a callback.
Callbacks receive the setup RNG; reproducible initialization requires a fresh RNG
with the same seed and callbacks that use that RNG.

## Reinitializing an existing shooting layer

The initialization helpers update tunable state-reset values after setup. Pass
`ps` and `st` explicitly; each helper returns the same `ps` and writes into its
existing buffers. Parameter values, fixed initial conditions, and the supplied
state are preserved. Omit `ps` and `st` to call `LuxCore.setup` automatically and
return the initialized `(ps, st)` pair. All keywords and hybrid strategy pairs are
forwarded to the explicit helper. These are setup operations, separate from differentiation
of the forward solve.

```julia
ps, st = forward_initialization(rng, layer)
trajectory, next_st = layer(nothing, ps, st)
```

| Helper | Guesses |
|---|---|
| `forward_initialization(rng, layer, ps, st)` | Incoming solution values at interior resets, using current parameter switches and fixed resets. |
| `linear_initialization(rng, layer, ps, st; u_infinity)` | Linear interpolation from the problem's initial state to `u_infinity`. Override the start with `u0`. |
| `constant_initialization(rng, layer, ps, st; u0)` | Repeat each variable's supplied value. |
| `custom_initialization(rng, layer, ps, st; u0s)` | Per-variable vectors, e.g. `Dict(1 => [0.5, 0.8])` for two resets of state 1. |
| `random_initialization(rng, layer, ps, st)` | Uniform samples within finite variable bounds. |
| `hybrid_initialization(rng, layer, ps, st, selectors...)` | Apply `variable_indices => helper` pairs sequentially. |

`variables` selects original state indices, for example `variables = [1, 3]`.
Every variable uses its own current reset grid from `st`. Custom vectors follow
that grid's value indices, as annotation defaults do. A dictionary passed to
`custom_initialization` selects its keys by default. Initial conditions stored
in Lux state through `Constant` are skipped by all helpers.

Forward initialization supports ODE problems and always solves sequentially,
including for layers configured with a parallel ensemble algorithm. It preserves
tunable assignments at the global start and fills selected interior and final
reset values from the incoming solution. Unselected resets remain active.
Other helpers also update selected global-start assignments. Generated values
must satisfy variable bounds; each helper validates all proposals before updating
`ps`. Forward initialization uses private working copies so the caller's state
and buffers remain unchanged if a solve or validation fails.

Use closures to combine strategies with different keywords:

```julia
hybrid_initialization(rng, layer, ps, st,
    1 => ((rng, layer, ps, st; variables) ->
        constant_initialization(rng, layer, ps, st; variables, u0 = Dict(1 => 0.5))),
    [2, 3] => forward_initialization,
)
```

Methods run in order: forward initialization sees guesses assigned by preceding
methods. A later failure does not undo earlier successful methods. These selectors
refer to variables, replacing the former API's segment selectors.
