# Getting started

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

The final state is 1: the two intervals contribute `0.25 × 1` and `0.75 × 1`.
This evaluates a prescribed control schedule; it does not optimize it.

## Next steps

Follow the [linear quadratic regulator tutorial](@ref linear_quad_reg) to add an
objective and solve an optimization problem. Use the [Manual](shooting.md) to
[define schedules](scheduling.md), [initialize shooting values](initialization.md),
[inspect trajectories](trajectories.md), or [add constraints](optimization_contracts.md).
For symbolic dynamics and costs, start with the
[ModelingToolkit tutorial](@ref mtk). Constructors and arguments are in the
[API reference](api.md).
