# Corleone.jl

Corleone turns differential equation solves into differentiable shooting layers
for dynamic optimization. It combines SciML solvers with Lux parameter and state
containers. You choose parameter schedules, state resets, an objective, and
constraints; the layer supplies trajectories and continuity residuals.

## Installation

For the introductory example, install Corleone, a solver, and LuxCore:

```julia
using Pkg
Pkg.add(["Corleone", "OrdinaryDiffEqTsit5", "LuxCore"])
```

Optimization and plotting packages are optional. Each tutorial lists its imports.
To use a local checkout, develop its path with `Pkg.develop(path = "path/to/Corleone.jl")`.

## A first shooting layer

Consider ``\dot{x}=p(t)`` with ``x(0)=0``. The parameter is a control: it is
``0.25`` before time 1 and ``0.75`` afterward. One switching time defines two
interval values.

```@example quickstart
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random

problem = ODEProblem((x, p, t) -> [p[1]], [0.0], (0.0, 2.0), [0.25])
control = Tunable(1; default = [0.25, 0.75],
    lowerbound = 0.0, upperbound = 1.0) => [1.0]
layer = ShootingLayer(problem, Tsit5(), control; controls = [1])
ps, st = LuxCore.setup(MersenneTwister(42), layer)
trajectory, next_st = layer(nothing, ps, st)

@assert isapprox(last(trajectory.u)[1], 1.0; atol = 1e-8) # hide
(; times = trajectory.t, states = round.(trajectory[:u₁]; digits = 6),
    controls = trajectory[:p₁])
```

`ps` contains tunable values; `st` contains schedules, caches, and fixed values.
The layer returns `(trajectory, next_st)`. The default numerical symbols are
`:u₁` for the first state and `:p₁` for the first parameter. State samples use the
saved solution grid; parameter histories use their own change grid.

An optimization model adds a scalar objective. Here it penalizes deviation from
a terminal target of 1.5 and retains the supplied parameters and state:

```@example quickstart
model = CorleoneDynamicOptProblem(layer,
    traj -> (last(traj.u)[1] - 1.5)^2; ps, st)
round(optimization_objective(model); digits = 6)
```

For a complete optimization, follow the
[linear quadratic regulator tutorial](@ref linear_quad_reg). For a symbolic model,
start with the [ModelingToolkit Lotka–Volterra tutorial](@ref mtk).

## Find the right guide

| Goal | Start here |
|---|---|
| Understand single and multiple shooting | [Shooting formulation](@ref shooting_formulation) |
| Query samples and interpret reset jumps | [Trajectories](@ref trajectory_guide) |
| Set initial guesses | [Initialization](initialization.md) |
| Supply bounds, objectives, and constraints | [Optimization contracts](optimization_contracts.md) |
| Define dynamics and costs symbolically | [ModelingToolkit](modelingtoolkit.md) |
| Choose a tested gradient configuration | [Automatic differentiation](automatic_differentiation.md) |
| Run independent shooting blocks concurrently | [Parallel shooting](parallel_shooting.md) |
| Look up constructors and fields | [API reference](api.md) |

This manual covers the root Corleone package. Independent packages under `lib`
and their tutorials are documented separately.
