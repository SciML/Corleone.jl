# Optimization-facing contracts

Corleone provides bounds and shooting-residual helpers independently of an optimizer
or vectorization package. Bounds follow generic Lux models and their parameter
structure; shooting-residual helpers apply to `ShootingLayer` and `Trajectory`.
The prepared facade below adds objectives and path constraints, with an optional
SciMLBase bridge with optional packing implementations. Optimizer block structure remains separate.

## Variable bounds

After `ps, st = LuxCore.setup(rng, layer)`, call:

```julia
lower = get_lower_bound(layer, ps, st)
upper = get_upper_bound(layer, ps, st)
# Convenience form:
lower, upper = get_bounds(layer, ps, st)
```

The directional functions follow Lux's recursive parameter convention:

- `AbstractLuxContainerLayer{fields}` wraps bounds under its declared child fields.
- `AbstractLuxWrapperLayer{field}` delegates to the wrapped layer without adding a key.
- NamedTuple, Tuple, and array collections recurse over their layers.
- Generic leaves default to `-Inf` or `Inf`, matching the supplied parameter shapes.

Empty wrapper parameter subtrees stay empty, including `FixedLayer` values stored
in state. This traversal requires no concrete bounds methods for `Parameter`,
`InitialConditions`, `PiecewiseConstant`, or `ParameterCollection`.

Only `ShootingLayer` has concrete layer specializations in Corleone. It first uses
the generic traversal to construct the parameter tree, then applies its cache's
per-symbol bounds. A larger Lux model can therefore contain ordinary layers and
wrapped shooting layers: ordinary leaves remain unbounded, while the shooting
subtree receives its symbolic bounds. Custom layers can extend `get_lower_bound`
and `get_upper_bound` when they need different bounds. The optional `st` argument
is forwarded to child layers.

The returned parameter trees have the native Lux parameter structure and the same
array shapes as `ps`. Each variable's bounds come from `layer.sys` and are repeated
across all its scheduled values. This includes `n + 1` values for a parameter with
`n` switching times. Unspecified bounds are `(-Inf, Inf)`.

`FixedLayer` values live in Lux state and produce empty parameter bounds. Pass-through
state variables also have no optimization entries. The helper reads metadata and
parameter shapes; it does not invoke initializers, consume RNG values, solve the
model, or change `ps` or `st`. Changes to the cache's bounds dictionary are reflected
on the next bounds query.

The overload `get_bounds(cache::ControlSymbolCache, symbol)` remains a bookkeeping
query returning one bound pair. `get_bounds(layer, ps, st)` expands that metadata
into optimizer variable bounds.

A vectorizer must apply the same ordering to parameters and both bound trees. For
example, with ComponentArrays:

```julia
q0 = ComponentArray(ps)
lower, upper = get_bounds(layer, ps, st)
lb, ub = ComponentArray(lower), ComponentArray(upper)
parameter_axes = getaxes(q0)
# Reconstruct numerical parameters inside the objective with
# ComponentArray(q, parameter_axes).
```

The core helper accepts either native NamedTuple parameters or a structured
ComponentArray, returning the same native bound trees for both, preserving the supplied field ordering. Flat vectors and
mismatched parameter field sets are rejected. The ComponentArrays extension only
adapts the parameter-tree representation; it does not build optimizer callbacks.
No implicit setup
or random parameter initialization occurs when requesting bounds.

## Shooting equality constraints

```julia
n = get_number_of_shooting_constraints(layer, ps, st)
trajectory, next_st = layer(nothing, ps, st)
residuals = shooting_constraints(trajectory)
```

The count is computed from schedules and `st.timestops` without a solve. State
residuals come from **tunable, active** state assignments, including assignments at the
problem start. Fixed assignments and assignments at the final endpoint contribute
none. After changing schedules, refresh `st.timestops` and recompute the count
before allocating optimizer buffers.

The residual vector starts with state residuals in chronological segment order,
then canonical state index order within each segment. Each value is the assigned
state minus the incoming state. Parameter matching residuals follow in canonical
parameter order, then increasing inserted interval index. Each inserted interval
contributes `values[i] - values[i - 1]` exactly once; original switches and fixed
parameters contribute none. It has length `n`, including the zero-constraint case. Equality-constraint
bounds are therefore `lcons = zeros(n)` and `ucons = zeros(n)`.

Use the allocating form for a differentiable vector-valued function:

```julia
residuals = shooting_constraints(trajectory)
```

Or write into an existing buffer or view:

```julia
shooting_constraints!(res, trajectory)
```

The bang form checks the exact buffer length before writing and returns the buffer.
It does not allocate an intermediate flattened residual vector. Its buffer must
support the residual element type; when computing ForwardDiff Jacobians, that
includes Dual numbers. Backends without array-mutation support should use the
allocating form. Scheduling/count metadata is non-differentiable; residual values
retain derivatives through assigned and incoming states.

## Parameter segments

To split tunable parameter intervals at interior full-state resets, call
`ps, st = add_parameter_segments!(layer, ps, st)` after setup and before creating
optimizer vectors or bounds. The helper reuses the layer and returns expanded
parameters and state, preserving their current numerical signal. Each new interval
starts with a duplicate of the previously active value. Original switches and
fixed parameters stay unchanged. Partial-state resets do not split parameters.

New right-hand interval indices are recorded in each parameter's
`st.parameter.parameters.<name>.lookup.inserted`. These markers distinguish
artificial splits from original switches. Repeated calls are idempotent. Changed
arrays and grids are newly allocated, affected lookup caches are cleared, and
`st.timestops` is refreshed from the current schedules. ComponentArray inputs
receive rebuilt axes. The layer's setup dimensions remain those of the original
layout; use the returned `ps` to size optimizer storage. Setup restores the original
layout.

Parameter matching residuals are computed on demand by `shooting_constraints` and
`shooting_constraints!` from the trajectory's stored layer, parameters, and state.
They are included in `get_number_of_shooting_constraints` but are not stored in
`trajectory.residuals`, which continues to hold the forward pass's state residuals.
Changing the stored parameter values changes the parameter differences on the next
constraint query; no additional solve is needed to compute those differences.

## Scope of the contract

The optimizer-facing mapping is deliberately explicit:

| Input | Contract |
|---|---|
| Initialized Lux parameters | Shape and ordering of optimization variables |
| Bounds metadata | `get_bounds(layer, ps, st)` |
| Cached schedules | `get_number_of_shooting_constraints(layer, ps, st)` |
| Solved trajectory | `shooting_constraints` or `shooting_constraints!` |
| Objective and path constraints | User callbacks operating on the trajectory |

These helpers do not add path/terminal constraints automatically, infer an objective
from quadrature metadata, or automatically split parameter intervals. Wrappers in
sibling packages can delegate to the underlying layer using their own nested
parameter/state locations. The prepared facade below uses these contracts without
changing their ordering or initialization behavior. Block structure belongs to a
future extension or package and is outside the core contract.

## Prepared optimization models

`CorleoneDynamicOptProblem` prepares objective and constraint callbacks for a
`ShootingLayer`. Supply initialized `ps` and `st` together to reuse their values,
including expanded parameter intervals. Omitting both calls `LuxCore.setup` once.
The model retains parameters and state by reference. It performs one initial
forward solve to determine the actual saved grid and constraint dimensions, then
releases that initial solution. This handles extra `saveat` samples and reset
boundaries without guessing their positions from the switching grid.

```jldoctest facade
julia> using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random

julia> prob = ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 2.0), [0.2];
           saveat = [0.5, 1.5]);

julia> layer = ShootingLayer(prob, Tsit5(),
           Tunable(1; kind = Val(:variable), default = 0.2,
               lowerbound = 0.0, upperbound = 2.0) => [1.0],
           Tunable(1; kind = Val(:parameter), default = 0.2,
               lowerbound = 0.0, upperbound = 1.0) => nothing);

julia> ps, st = LuxCore.setup(MersenneTwister(1), layer);

julia> ps, st = add_parameter_segments!(layer, ps, st);

julia> model = CorleoneDynamicOptProblem(layer,
           traj -> (last(traj.u)[1] - 1)^2,
           :u₁ => (; t = [0.5, 1.0, 1.5, 2.0], bounds = (0.0, 1.5)); ps, st);

julia> optimization_objective(model) ≈ 0.36
true

julia> length(optimization_constraints(model)) # Four path values and two shooting equalities.
6
```

### Objectives and constraints

An objective callback receives the trajectory and returns a real scalar. A symbol,
integer state index, or symbolic expression instead selects its terminal value.
Omitting `loss` sums the terminal values of the quadrature symbols declared on the
layer; it does not sum quadrature values across segments. Without quadrature
metadata, an explicit loss is required.

Declare each constraint as `selector => (; t, bounds, side)`. Definitions retain
user order. Each selected time contributes values in selector order, followed by
all state and parameter shooting equalities. Scalar limits are repeated; vector
limits must match the resulting number of scalar entries. Reversed limits, NaN
bounds and dimension mismatches are rejected.

| Definition | Meaning |
|---|---|
| `:u₁ => (; bounds = (0, 1))` | Terminal state constraint. |
| `:u₁ => (; t = :all, bounds = (0, 1))` | Constraint at every saved time. |
| `:(u₁ + p₁) => (; t = [1.0, 2.0], bounds = (0, 3))` | Mixed state/parameter expression at specified saved times. |
| `:u₁ => (; t = 1.0, bounds = (0, 1), side = :pre)` | First saved sample at that time. |
| `:u₁ => (; t = 1.0, bounds = (0, 1), side = :both)` | Every saved sample at that time, in saved order. |
| `(traj -> last(traj.u)[1]^2) => (; bounds = (0, 4))` | Whole-trajectory callback returning a scalar or flat vector. |
| `(traj -> first.(traj.u)) => (; t = [1.0], bounds = (0, 1))` | Select from a callback returning one entry per saved sample. |

The default `side = :post` selects the last sample at a requested time, which is
the outgoing state at an interior reset. `:pre` chooses the first and `:both`
retains both. If only one sample exists, all choices select that sample; there is
no implicit incoming sample at the problem start. Parameter values are queried
from the parameter layer and remain right-continuous regardless of state-side
selection. Parameter selectors therefore work on the state sample grid rather
than incorrectly indexing the shorter parameter-knot history.

Requested times must be present exactly; trajectories do not interpolate. Include
missing points in the ODE problem's `saveat` before preparing the model. Changing
saved layout or shooting-constraint count requires reconstructing the model, as
does changing callback output dimensions. Reuse the same shooting layer and new
parameters/state when doing so.

`optimization_constraints` is allocating and differentiable. The bang form checks
buffer length before solving or writing, evaluates user constraints, then writes
shooting residuals directly into the remaining buffer. Use the allocating form
for reverse-mode backends that do not support mutation. Each objective or
constraint callback performs one prediction; there is no numerical solution cache
shared across calls or AD contexts.

### SciMLBase optimization bridge

SciMLBase provides the bridge constructors; ComponentArrays supplies the default
packing implementation. Optimization.jl and a solver are needed to run the optimization.
For example, with ADTypes/ForwardDiff and an installed optimizer:

```julia
using Optimization, ComponentArrays, ADTypes, ForwardDiff
using OptimizationMOI, Ipopt

optprob = OptimizationProblem(model, AutoForwardDiff(); vectorize = default_vectorizer())
result = solve(optprob, Ipopt.Optimizer();
    hessian_approximation = "limited-memory", tol = 1e-8)

_, reconstruct = pack(Val(:ComponentArrays), model.ps)
parameters = reconstruct(result.u)
trajectory, state = model.layer(nothing, parameters, optprob.p)
```

`optprob.u0` is a flat numerical vector; bounds are flattened in the same order.
Packing a supplied ComponentVector reuses its backing storage, while packing a
native NamedTuple allocates optimizer storage once. The reconstruction callable
retains the layout separately from the numerical vector. During evaluations it
wraps the **current** optimizer vector using those axes; it does not read stale initial
parameter values. `optprob.p` holds the supplied Lux state. Fixed values remain
outside the optimization variables. Constraint bounds combine user limits with
zero bounds for all shooting equalities. With no constraints, `cons`, `lcons` and
`ucons` are `nothing`.

`OptimizationFunction(model, adtype; inplace = true)` constructs the callbacks
without repeating setup. Pass `inplace = false` for allocating constraints and the
out-of-place derivative convention. Derivative options such as custom Jacobians
belong in `function_kwargs = (; cons_j = ...)` when constructing the problem;
remaining keywords are passed to `OptimizationProblem`. The generated `cons`
callback cannot be replaced: declare constraints on the model instead.

The convenience form `OptimizationProblem(layer, adtype; loss, constraints, ps, st)`
prepares the model and bridge together. The `vectorize` keyword selects packing:
`OptimizationProblem(model, adtype; vectorize = Val(:ComponentArrays))`. Its default,
`default_vectorizer()`, selects a supported loaded extension, currently
ComponentArrays. If none is loaded it throws `UndefKeywordError(:vectorize)`; load
ComponentArrays or pass an explicit implementation. Other tags can implement `pack`
without changing the bridge. The same keyword applies to OptimizationFunction.
The bridge does not add new AD backend capabilities; see the backend requirements
in [Automatic differentiation](automatic_differentiation.md).

### Parameter packing interface

```julia
tag = Val(:ComponentArrays)
q0, reconstruct = pack(tag, ps)   # Flat numerical vector and a callable layout.
parameters = reconstruct(q0)     # ComponentArray viewing q0.
```

Only packing depends on ComponentArrays. The objective, constraint, and SciMLBase
constructors use the callable returned by `pack`. Other implementations may return
a reconstruction function or callable struct producing a NamedTuple or another
container with the named property access expected by the layer.

The ComponentArrays implementation returns the underlying numerical array and a
callable retaining only the axes and expected length. Consequently, optimizers and
AD backends can pass plain vectors without losing the parameter layout. Calling
`reconstruct(q)` wraps the current vector in a ComponentArray without copying its
values. It preserves the current scalar type, including tracked values and Dual
numbers. No NamedTuple conversion is required. Packing a preexisting ComponentVector
reuses its backing storage; packing a NamedTuple allocates flat storage once.

Each packing implementation must use the same ordering for parameters and both
bound trees. Reconstructors must use the supplied vector's values, rather than
retaining initial numerical parameter values.
