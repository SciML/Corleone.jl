# [Initializing models](@id Initializing-shooting-variables)

## Annotation defaults

Set `default` on `Tunable` or `Constant` to initialize values during `LuxCore.setup`.
The [scheduling guide](scheduling.md) explains selectors and interval counts.

| Default | Values supplied |
|---|---|
| Scalar | Repeated at every scheduled entry |
| `AbstractVector` | Entries in schedule order |
| `(rng, n::Int) -> values` | A vector of `n` values |
| `(rng, times::AbstractVector) -> values` | A vector evaluated at initialization times |

State defaults have one value per reset, in supplied grid order. Parameter defaults
have one value per interval; time callbacks receive `[tspan[1]; switching_times]`.
A parameter with a `nothing` grid receives `[tspan[1]]`; a state with an empty or
`nothing` grid has no reset values.

```@example initialize
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random

problem = ODEProblem((x, p, t) -> [p[1]], [0.0], (0.0, 3.0), [1.0])
resets = Tunable(1; kind = Val(:variable),
    default = (rng, times::AbstractVector) -> 10 .* times) => [1.0, 2.0]
layer = ShootingLayer(problem, Tsit5(), resets, Constant(1) => nothing)
rng = MersenneTwister(42)
ps, st = LuxCore.setup(rng, layer)
trajectory, _ = layer(nothing, ps, st)
trajectory[:u₁]
```

The reset guesses are 10 and 20. For guesses from a reference solution, use a
time callback that samples that solution; Corleone's `Trajectory` does not interpolate.

Callbacks must return numeric vectors of the required length within annotation
bounds. The count signature wins if both are applicable: annotate `times` with
`::AbstractVector` to request times. Exceptions propagate without retrying a
callback. Scalar/vector defaults are checked during construction, and callback
outputs during setup.

Supplied vectors and callback results are retained by reference. Return a fresh
array from a callback for independent repeated setup calls. Reproducibility requires
a fresh seeded RNG and callbacks that use it.

## Improve shooting guesses

The initialization helpers update existing **tunable state-reset buffers**, leaving
parameters and fixed initial conditions unchanged. Pass `ps` and `st` explicitly
to mutate those buffers and return the same `ps`; omit both to run setup and
return `(ps, st)`. These are setup operations, separate from forward differentiation.

```@example initialize
forward_initialization(rng, layer, ps, st)
trajectory, next_st = layer(nothing, ps, st)
@assert maximum(abs, shooting_constraints(trajectory)) < 1e-8 # hide
round.(trajectory[:u₁]; digits = 6)
```

Forward initialization replaces the interior guesses with incoming solution values,
here 1 and 2, making the initial trajectory continuous.

| Helper | Strategy |
|---|---|
| `forward_initialization` | Solve sequentially with current parameters and fixed resets. |
| `linear_initialization` | Interpolate from `u0` (the problem initial state by default) to required `u_infinity`. |
| `constant_initialization` | Repeat supplied `u0` values. |
| `custom_initialization` | Use supplied per-variable vectors `u0s`. |
| `random_initialization` | Sample uniformly within finite variable bounds. |
| `hybrid_initialization` | Apply `variable_indices => helper` strategies in order. |

Use `variables = [1, 3]` to select original state indices. Custom vectors follow
each state's current grid. A dictionary of custom guesses selects its keys by
default. Endpoints and custom guesses also accept `index => value` pairs; pass
`variables` explicitly to select a subset. See the
[helper docstrings](api.md#Shooting-layers) for accepted argument forms.

Forward initialization supports ODE problems and runs sequentially even for a
parallel layer. It preserves tunable global-start assignments, fills selected
interior and final reset values, and keeps unselected resets active. Other helpers
also update selected global-start values. Each helper validates all its proposals
before writing; a failed forward solve leaves caller buffers unchanged. Hybrid
methods run in order, and a later failure does not undo earlier successful methods.
