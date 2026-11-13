# [Trajectories](@id trajectory_guide)

Calling a [`ShootingLayer`](@ref) returns `(trajectory, next_st)`. The
[`Trajectory`](@ref) holds the saved state samples and gives access to the
parameter values used in the solve. It stores samples only; it does not interpolate
between them.

## Reading states and parameters

Numerical problems name their states `:u₁`, `:u₂`, … and their parameters `:p₁`,
`:p₂`, …. Problems with a symbolic system, for example from ModelingToolkit, keep
their own names.

| Query | Result |
|---|---|
| `trajectory.t[i]`, `trajectory.u[i]` | The time and state vector of saved sample `i`. |
| `trajectory[state_symbol]` | The state's values at all saved times. |
| `trajectory[parameter_symbol]` | The parameter's values at its own switching times. |
| `parameter_values_at_time(trajectory, trajectory, t)` | The full parameter vector at time `t`. |

The last query comes from SymbolicIndexingInterface. Parameters are piecewise
constant and right-continuous, and their switching times usually differ from the
state sample times, so don't pair the two series by index.

Read `trajectory.t` and `trajectory.u` together rather than predicting the saved
grid from switching times: solver options such as `saveat` add samples, and not
every integration stop is saved. For values at specific times in an objective, use
[timed expressions](@ref timed_expressions).

## Both sides of a reset

At an interior state reset the trajectory keeps two samples with the same time: the
incoming endpoint and the outgoing start. `reset_indices` points to the
**outgoing** samples, so the incoming value sits just before each one.

```@example reset_samples
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random

problem = ODEProblem((x, p, t) -> [p[1]], [0.0], (0.0, 3.0), [1.0])
layer = ShootingLayer(problem, Tsit5(),
    Tunable(1; kind = Val(:variable), default = [0.5, 1.5]) => [1.0, 2.0],
    Constant(1) => nothing)
ps, st = LuxCore.setup(MersenneTwister(42), layer)
trajectory, next_st = layer(nothing, ps, st)
(; times = trajectory.t, outgoing_indices = trajectory.reset_indices)
```

The saved times are `[0, 1, 1, 2, 2, 3]`. At time 1 the incoming value is about 1
and the assigned value is 0.5:

```@example reset_samples
k = first(trajectory.reset_indices)
@assert isapprox(trajectory.u[k - 1][1], 1.0; atol = 1e-8) # hide
@assert trajectory.u[k][1] == 0.5 # hide
round.([trajectory.u[k - 1][1], trajectory.u[k][1]]; digits = 6)
```

[`shooting_constraints`](@ref) returns the matching residuals, with the sign defined
in [multiple shooting](@ref shooting_formulation):

```@example reset_samples
residuals = shooting_constraints(trajectory)
@assert isapprox(residuals, [-0.5, 0.0]; atol = 1e-8) # hide
round.(residuals; digits = 6)
```

The raw field `trajectory.residuals` stores these residuals per segment, for the
tunable states only; prefer `shooting_constraints`, which also includes
inserted-parameter equalities. The [plotting recipe](@ref trajectoryplot) draws each
segment separately, so jumps appear as gaps rather than vertical lines.

## [Shared data and differentiation](@id trajectory_ownership)

A trajectory keeps references to the layer, the parameters, and the returned Lux
state, and its state arrays are shared with the segment solutions. Parameter values
are looked up on demand from those references.

In practice this means:

- Changing `ps` afterwards changes what parameter queries return, but not the saved
  states. Run the layer again to get a consistent trajectory.
- Don't mutate any of this data between a forward call and its pullback.

`symbolic_container(trajectory)` returns the layer's [`ControlSymbolCache`](@ref),
which answers control and quadrature queries; see [symbolic metadata](@ref symbolic_metadata_guide).

Derivatives are supported with respect to numerical values on a fixed time grid, not
with respect to reset or switching times. See
[automatic differentiation](@ref automatic_differentiation) for backends.
