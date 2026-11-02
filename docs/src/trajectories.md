# [Trajectories and reset samples](@id trajectory_guide)

[`ShootingLayer`](@ref) returns `(trajectory, next_st)`. A [`Trajectory`](@ref)
contains saved state samples and access to the parameters used for prediction.
It does not interpolate or integrate between those samples.

## Saved grids and both sides of a reset

Continuous handoffs appear once in `trajectory.t`. A scheduled interior reset
retains both the incoming endpoint and outgoing start, even if their values are
equal. `reset_indices` points to the **outgoing** samples. The immediately
preceding samples contain the incoming values.

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

The saved times are `[0, 1, 1, 2, 2, 3]`. The two states at time 1 are the
incoming value of approximately 1 and the assigned value 0.5:

```@example reset_samples
k = first(trajectory.reset_indices)
@assert isapprox(trajectory.u[k - 1][1], 1.0; atol = 1e-8) # hide
@assert trajectory.u[k][1] == 0.5 # hide
round.([trajectory.u[k - 1][1], trajectory.u[k][1]]; digits = 6)
```

`trajectory.residuals` stores state matching residual vectors per integration
segment start. Their entries are compressed to the active tunable variables;
they are not a matrix indexed by every state and saved time. Use
[`shooting_constraints`](@ref) for the complete optimizer residual vector,
including any inserted-parameter equalities:

```@example reset_samples
residuals = shooting_constraints(trajectory)
@assert isapprox(residuals, [-0.5, 0.0]; atol = 1e-8) # hide
round.(residuals; digits = 6)
```

The [shooting formulation](@ref shooting_formulation) defines the residual sign.
The [plotting API](@ref trajectoryplot) draws reset segments without a continuous
line across each jump and can show variable matching residuals.

## Symbolic state and parameter queries

For an ordinary numerical problem, the generated state and parameter names are
`:u₁`, `:u₂`, … and `:p₁`, `:p₂`, …. Problems with a symbolic container retain its
original symbols and indices.

| Query | Result |
|---|---|
| `trajectory.u[i]` | State vector at saved index `i`. |
| `trajectory.t[i]` | Time associated with that state vector. |
| `trajectory[state_symbol]` | State values on the saved trajectory grid, including both reset sides. |
| `trajectory[parameter_symbol]` | Parameter values on the parameter-change grid. |
| `parameter_values_at_time(trajectory, trajectory, t)` | Full parameter vector at time `t`, for SII callers. |

The last query is from SymbolicIndexingInterface. Parameters are piecewise
constant and right-continuous, and their history grid can differ from the state
sample grid. Do not combine the two histories by assuming matching column indices.

Timed numerical expressions such as `:(u₁(1.0))` select the post-reset state by
default. A callback can inspect either saved side explicitly. Bare signal names
inside expressions denote their full series on the saved grid; a plain Symbol
objective selector denotes the terminal value. See
[expression evaluation](optimization_contracts.md#Timed-Julia-expressions).

Internal solver saving options, including `saveat`, can add state samples.
Integration stops need not each be saved. Read the actual `trajectory.t` and
`trajectory.u` together rather than reconstructing a saved grid from control
switches or segment endpoints.

## Ownership and differentiation

The trajectory retains the layer, supplied parameters, and returned Lux state by
reference. Its numerical state arrays are shared with the segment solutions.
Parameter histories are evaluated on demand through the retained parameter layer;
there is no detached parameter-history copy.

Consequently, changing a parameter buffer changes later parameter queries, but
does not recompute saved state values. Run the layer again for a trajectory
consistent with changed parameters. Avoid mutating retained numerical data between
a forward evaluation and its pullback.

Symbolic metadata is shared through `symbolic_container(trajectory)`, which returns
the layer's [`ControlSymbolCache`](@ref). Query control and quadrature subsets on
that wrapper; see [symbolic bookkeeping](symbolic_metadata.md).

Differentiation is validated for numerical values on fixed time grids. It does
not establish derivatives through changing reset times, grid topology, or event
schedules. Backend and sensitivity requirements are documented in
[automatic differentiation](automatic_differentiation.md).
