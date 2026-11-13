# [Initial guesses](@id initialization_guide)

Good starting values matter, especially for multiple shooting. Corleone offers two
ways to set them: annotation defaults, which are applied during `LuxCore.setup`,
and initialization helpers, which improve the shooting values of an existing setup.

## Annotation defaults

Set `default` on a `Tunable` or `Constant` annotation. It can be a value, a vector,
or a function:

| Default | Values supplied |
|---|---|
| Scalar | The same value for every interval or reset. |
| `AbstractVector` | One entry per interval or reset, in grid order. |
| `(rng, n::Int) -> values` | A vector of `n` values, for example random guesses. |
| `(rng, times::AbstractVector) -> values` | One value per time, computed from the times. |

Parameters have one value per interval; a time callback receives the interval start
times `[tspan[1]; switching_times]`. States have one value per reset, at the reset
times. In this example, each shooting value is initialized to ten times its reset
time:

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

The reset guesses are 10 and 20. To start from a reference solution, sample it in a
time callback; a `Trajectory` itself does not interpolate.

!!! details "Callback rules"
    - Callbacks must return a numeric vector of the right length, within the
      annotation's bounds. Scalar and vector defaults are checked when the layer is
      built, callback results during setup.
    - Annotate the argument as `times::AbstractVector` to get times; otherwise the
      count signature is used.
    - Results are stored by reference, so return a fresh array if you call setup
      repeatedly. For reproducible guesses, use a seeded RNG inside the callback.

## Improving shooting guesses

The initialization helpers overwrite the **tunable state-reset values** of an
existing setup, leaving parameters and fixed values untouched. The most useful one
is `forward_initialization`, which solves the problem once with the current
parameters and copies the incoming states into the shooting values:

```@example initialize
forward_initialization(rng, layer, ps, st)
trajectory, next_st = layer(nothing, ps, st)
@assert maximum(abs, shooting_constraints(trajectory)) < 1e-8 # hide
round.(trajectory[:u₁]; digits = 6)
```

The guesses are now 1 and 2, and the initial trajectory is continuous.

| Helper | Strategy |
|---|---|
| `forward_initialization` | Solve forward with the current parameters. |
| `linear_initialization` | Interpolate linearly from `u0` (the problem's initial state by default) to the required `u_infinity`. |
| `constant_initialization` | Repeat the given `u0` values. |
| `custom_initialization` | Use the given per-variable vectors `u0s`. |
| `random_initialization` | Sample uniformly within finite bounds. |
| `hybrid_initialization` | Apply `variable_indices => helper` pairs in order. |

Passing `ps` and `st` updates them in place and returns `ps`; leaving both out runs
setup and returns `(ps, st)`. Use `variables = [1, 3]` to restrict a helper to some
states. Custom vectors and dictionaries follow each state's grid, and endpoints
accept `index => value` pairs. The [helper docstrings](@ref core_api) list all
accepted forms.

A few details are worth knowing:

- `forward_initialization` supports ODE problems and always runs sequentially,
  even for a parallel layer. It keeps tunable initial values and fills interior
  and final resets.
- The other helpers also update tunable initial values of the selected states.
- Each helper checks all its values before writing anything, so a failed forward
  solve leaves `ps` unchanged. In a hybrid strategy, earlier successful steps are
  kept if a later one fails.
