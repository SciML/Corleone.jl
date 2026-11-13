# Getting started

## Installation

Install Corleone together with an ODE solver and LuxCore:

```julia
using Pkg
Pkg.add(["Corleone", "OrdinaryDiffEqTsit5", "LuxCore"])
```

To optimize, also add Optimization.jl, a solver, an AD backend, and
ComponentArrays. This page uses Ipopt:

```julia
Pkg.add(["Optimization", "OptimizationMOI", "Ipopt", "ForwardDiff", "ComponentArrays"])
```

Each tutorial lists the packages it uses. To work on a local checkout instead, run
`Pkg.develop(path = "path/to/Corleone.jl")`.

## The core workflow

Every Corleone problem follows the same steps:

1. Write the dynamics as an ordinary SciML `ODEProblem`.
2. Annotate the parameters and states that should be optimized, and give each one a
   time grid. An annotation paired with a grid, `Tunable(i) => grid`, is a schedule.
3. Build a [`ShootingLayer`](@ref) and call `LuxCore.setup` to get the tunable values
   `ps` and the fixed data `st`.
4. Evaluate the layer to get a [`Trajectory`](@ref), or wrap it in a
   [`CorleoneDynamicOptProblem`](@ref) with an objective and constraints.
5. Convert the model to an `OptimizationProblem`, solve it, and map the result back
   to the structured parameters.

The figure shows this loop. The layer integrates the dynamics segment by segment
for the current parameters, the objective and constraints are evaluated on the
resulting trajectory, and the optimizer updates `ps` using derivatives from AD.

![The optimization loop: a ShootingLayer maps ps and st to a trajectory with a piecewise-constant control p(t) and a state x(t) that is reset at t₂, leaving a residual r₂; the objective, constraints, and residuals are differentiated with SciMLSensitivity.jl, and Optimization.jl updates ps.](assets/shooting-sequence.svg)

The rest of this page walks through these steps on a one-state example.

## A first shooting layer

Consider ``\dot{x}=p(t)`` with ``x(0)=0`` on ``[0, 2]``. We let the parameter ``p``
switch once, at ``t = 1``, so it is a control with two interval values. We start
with ``0.25`` on the first interval and ``0.75`` on the second.

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

`Tunable(1; ...) => [1.0]` makes the first parameter tunable and lets it switch at
``t = 1``. `ps` holds the values an optimizer may change; `st` holds the schedules,
caches, and anything marked `Constant`. Numerical problems name their states `:u₁`,
`:u₂`, … and their parameters `:p₁`, `:p₂`, …, and the trajectory can be indexed
with these symbols.

The final state is ``0.25 \cdot 1 + 0.75 \cdot 1 = 1``.

## Optimizing the control

Now we choose the two control values so that the state passes through ``0.5`` at
``t = 1`` and reaches ``1.5`` at ``t = 2``. The objective is a Julia expression, and
`u₁(t)` reads the state at time `t`:

```@example quickstart
using ComponentArrays, ForwardDiff, Optimization, OptimizationMOI, Ipopt

model = CorleoneDynamicOptProblem(layer,
    :((u₁(1.0) - 0.5)^2 + (u₁(2.0) - 1.5)^2); ps, st)
optprob = OptimizationProblem(model, AutoForwardDiff())
result = solve(optprob, Ipopt.Optimizer(); tol = 1e-12, print_level = 0)

_, reconstruct = pack(default_vectorizer(), ps)
optimized, _ = layer(nothing, reconstruct(result.u), st)
@assert isapprox(optimized[:p₁], [0.5, 1.0]; atol = 1e-5) # hide
(; retcode = result.retcode, controls = round.(optimized[:p₁]; digits = 6))
```

The optimizer finds the controls ``0.5`` and ``1.0``. `pack` turns the structured
parameters into the flat vector the optimizer works with, and `reconstruct` maps the
solution back so we can evaluate the layer again.

## Next steps

- The [linear quadratic regulator tutorial](@ref linear_quad_reg) adds an integral
  cost and a fine control grid.
- The Manual explains each building block:
  [parameter schedules](scheduling.md), [single and multiple shooting](shooting.md),
  [initial guesses](initialization.md), [trajectories](trajectories.md), and
  [objectives and constraints](optimization_contracts.md).
- For symbolic dynamics and costs, start with the [ModelingToolkit tutorial](@ref mtk).
- To fit parameters to data, see [CorleoneInverse](inverse/index.md); to design
  experiments, see [CorleoneOED](oed/index.md).
