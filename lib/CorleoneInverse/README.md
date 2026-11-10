# CorleoneInverse

Least-squares parameter estimation for Corleone's shooting layers. An `Experiment`
stores its observations per variable as `selector => (t, y[, σ])` pairs; a shooting
layer predicts it with `layer(experiment, ps, st)`. Several experiments share the
layer's tunable parameters, whose transforms and priors are declared on Corleone's
annotations. Fits run through NonlinearSolve or Optimization.

```julia
using CorleoneInverse, OrdinaryDiffEqTsit5, ComponentArrays, NonlinearSolve, LuxCore, Random
using TransformVariables: asℝ₊

problem = ODEProblem((u, p, t) -> -p[1] .* u, [1.0], (0.0, 2.0), [0.5])
layer = ShootingLayer(problem, Tsit5(), Tunable(1; kind = Val(:parameter), transform = asℝ₊) => nothing)
t = collect(0.2:0.2:2.0)
experiment = Experiment(:u₁ => (t, exp.(-1.3 .* t), 0.01))
ps, st = LuxCore.setup(Xoshiro(1), layer)
sol = solve(NonlinearLeastSquaresProblem(layer, experiment, ps, st), LevenbergMarquardt())
ParameterEstimate(layer, experiment, sol.u, ps, st)
```

Use `OptimizationProblem(layer, experiments, ps, st, ad)` for bounds, general priors,
and multiple shooting. See the [manual](docs/src/index.md) and the [API reference](docs/src/api.md).

`Experiments(problem, shared_annotations...; experiments=(name=(annotations=(...),
observations=data), ...))` defines shared and individual ordinary parameters,
controls, and initial variables without a solver. Lux setup initializes each owned
value once; calling the container returns merged `(ps, st)` tuples. Execute with
`shooting(experiments, ps, st)` for a vector of trajectories and updated state, or
fit with `NonlinearLeastSquaresProblem(shooting, experiments, ps, st)`. See
[`examples/experiment_layers/main.jl`](examples/experiment_layers/main.jl).
