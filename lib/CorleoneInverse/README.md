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
model = CorleoneDynamicOptProblem(layer, experiment; ps, st)
sol = solve(NonlinearLeastSquaresProblem(model), LevenbergMarquardt())
ParameterEstimate(model, sol.u)
```

Least squares supports parameter bounds and untransformed Normal priors.
Use `OptimizationProblem(model, ad)` for general priors and constraints.
Declare constraints on `CorleoneDynamicOptProblem(shooting, experiments, constraints...;
ps, st, lcons, ucons)`. Scalar measurement expressions such as `:(p₁ * u₁)` compile
once on each experiment's observation grid. See the [manual](docs/src/index.md), with guides to [observations](docs/src/observations.md) and [multiple experiments](docs/src/experiments.md), and the [API reference](docs/src/api.md).

`Experiments(problem, shared_annotations...; experiments=(name=(annotations=(...),
observations=data), ...))` defines shared model parameters, experiment-local
controls, and shared or local initial variables without a solver. Lux setup initializes each owned
value once; calling the container returns merged `(ps, st)` tuples. Execute with
`shooting(experiments, ps, st)` for a vector of trajectories and updated state, or
prepare `model = CorleoneDynamicOptProblem(shooting, experiments; ps, st)` and
fit with `NonlinearLeastSquaresProblem(model)`. See
[`examples/experiment_layers/main.jl`](examples/experiment_layers/main.jl).

The [example collection](examples/README.md) includes seven adapted tutorials from
Christoph Plate ([@chplate](https://github.com/chplate))'s
[PR #168](https://github.com/SciML/Corleone.jl/pull/168), covering joint experiments,
fixed OED controls, multiple shooting, SIR dynamics, a stiff DAE, and a Lux UDE.
