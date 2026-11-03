# Scheduling parameters

Pair a `Tunable` or `Constant` annotation with a grid to select which values vary
and when. Controls are entries of the differential equation's parameter vector;
`controls` marks them for symbolic queries and plots without changing their storage.

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

## Interval values and storage

A parameter grid contains **switching times**: `n` switches require `n + 1` values.
Supply switching times in sorted order. At a switch, the right-hand interval is
active. Use `nothing` for one value over the whole horizon. Different parameters
may have different grids.

`Tunable` values live in `ps`; `Constant` values live in Lux state `st`, alongside
schedules and caches. Unspecified numerical parameters receive one tunable value
initialized from `problem.p`. Mark physical parameters `Constant` when they must
remain outside the optimizer vector. Unspecified states pass through their incoming
values; state annotations instead schedule [resets](shooting.md).

Select by original index or symbol. A vector selector takes one default per
variable; a vector of vectors supplies each variable's time series:

```julia
using Corleone

# Two parameters, each with two interval values.
controls = Tunable([1, 2]; default = [[0.2, 0.3], [0.5, 0.6]]) => [1.0]
```

Equal grids share a lookup, including across separate annotations. Each member
still stores a vector of scalar values; fixed and tunable members may share a
grid. Changing that grid affects all its members. Reconstruct the layer to move
only some members to another grid. Avoid depending on private nesting paths;
use [parameter packing](optimization_contracts.md#Parameter-packing-interface)
to reconstruct an optimizer result.

## Defaults and schedule changes

Set initial guesses with annotation defaults or callbacks; see
[initialization](initialization.md). Bounds constrain stored tunable values,
not the state between samples; see [constructing shooting problems](shooting.md).

Setup caches the union of endpoints, switches, and resets in `st.timestops`.
After changing schedules in state, refresh it before evaluating:

```julia
st = merge(st, (; timestops = Corleone.get_timestops(layer, nothing, st)))
```

Changing a grid's size also requires matching value buffers. Reprepare optimization
models after schedule or layout changes. [`add_parameter_segments!`](@ref) is the
supported helper for splitting intervals at full-state shooting boundaries while
preserving their current values; see [shared parameters](shooting.md#Shared-parameters-and-inserted-intervals).
