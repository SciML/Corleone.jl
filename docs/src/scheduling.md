# [Scheduling parameters](@id scheduling_guide)

A schedule says which values the optimizer may change and when they may change.
You write one by pairing an annotation with a time grid:

- `Tunable(i) => grid` makes parameter `i` an optimization variable that switches
  at the times in `grid`.
- `Constant(i) => grid` keeps parameter `i` fixed, but still lets it follow a
  prescribed piecewise-constant profile.
- `grid = nothing` means a single value over the whole horizon.

In this example, the first parameter is a fixed decay rate and the second is a
tunable input that switches once:

```@example schedules
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random

problem = ODEProblem((x, p, t) -> [p[1] * x[1] + p[2]],
    [1.0], (0.0, 2.0), [-1.0, 0.0])
layer = ShootingLayer(problem, Tsit5(),
    Constant(1) => nothing,
    Tunable(2; default = [0.25, 0.75], lowerbound = 0.0, upperbound = 1.0) => [1.0];
    controls = [2])
ps, st = LuxCore.setup(MersenneTwister(42), layer)
trajectory, next_st = layer(nothing, ps, st)
trajectory[:p₂]
```

Controls are ordinary entries of the parameter vector. The `controls = [2]`
keyword only labels parameter 2 as a control for plots and symbolic queries; it
does not change how the value is stored.

## Interval values and storage

A grid lists **switching times**, so `n` switches give `n + 1` interval values.
Supply the times in sorted order. At a switching time, the new value is already
active. Each parameter can have its own grid.

`Tunable` values live in `ps`, the vector the optimizer changes. `Constant` values
live in the Lux state `st`, together with the schedules and caches. Parameters you
don't annotate get one tunable value initialized from `problem.p`, so mark physical
constants as `Constant` to keep them out of the optimization.

You can select several parameters at once by index or by symbol. Give one default
per parameter, or one vector of interval values per parameter:

```julia
using Corleone

# Two parameters, each with two interval values.
controls = Tunable([1, 2]; default = [[0.2, 0.3], [0.5, 0.6]]) => [1.0]
```

Parameters with identical grids share one lookup table, even across separate
annotations, but each keeps its own values. To read the values in a stable layout,
for example after optimization, use
[parameter packing](@ref parameter_packing) rather than the nested fields of `ps`.

State annotations work the same way, but they schedule [resets](@ref shooting_formulation)
of the state instead of parameter switches.

## Initial values and bounds

Set initial guesses through the annotation's `default`, as described in
[initial guesses](@ref initialization_guide). `lowerbound` and
`upperbound` constrain the stored interval values; see
[bounds and accuracy](@ref shooting_bounds) for what they do and don't guarantee.

## [Changing a schedule after setup](@id schedule_changes)

The easiest way to change a grid is to build a new layer. If you do modify the
schedules stored in `st`, refresh the cached integration grid before evaluating:

```julia
st = merge(st, (; timestops = Corleone.get_timestops(layer, nothing, st)))
```

Changing the number of intervals also requires value buffers of the new size, and
any prepared optimization model must be rebuilt. Since grids are shared between
parameters, a change affects every member of that grid.

[`add_parameter_segments!`](@ref) is the supported way to split parameter intervals
at multiple-shooting boundaries while keeping their current values; see
[shared parameters](@ref shooting_parameter_segments).
