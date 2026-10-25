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
