# Optimization-facing contracts

Corleone provides bounds and shooting-residual helpers independently of an optimizer
or vectorization package. Bounds follow generic Lux models and their parameter
structure; shooting-residual helpers apply to `ShootingLayer` and `Trajectory`. They do not construct objectives or path constraints,
flatten parameters, or provide optimizer block structure.

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

The count is computed from schedules and `st.timestops` without a solve. Only
**tunable, active** state assignments contribute, including assignments at the
problem start. Fixed assignments and assignments at the final endpoint contribute
none. After changing schedules, refresh `st.timestops` and recompute the count
before allocating optimizer buffers.

The residual vector uses chronological segment order, then canonical state index
order within each segment. Each value is the assigned state minus the incoming
state. It has length `n`, including the zero-constraint case. Equality-constraint
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
from quadrature metadata, or introduce per-block parameter copies. Wrappers in
sibling packages can delegate to the underlying layer using their own nested
parameter/state locations. Automatic objective/path-constraint construction and
an OptimizationProblem bridge are separate follow-up work. Block structure belongs
to a future extension or package and is outside the core contract.
