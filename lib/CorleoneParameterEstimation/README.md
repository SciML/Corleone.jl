# CorleoneParameterEstimation.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Global Docs](https://img.shields.io/badge/docs-SciML-blue.svg)](https://docs.sciml.ai/Corleone/stable/)

[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor%27s%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)

CorleoneParameterEstimation.jl is a component of the
[Corleone.jl](https://github.com/SciML/Corleone.jl) monorepo. It provides parameter estimation
for ordinary differential equation models given time-series observations. Given a Corleone
shooting layer (built from an `ODEProblem`), an observation map and measured data, it sets up
a nonlinear *least-squares* problem and solves it with the standard,
SciML least-squares solvers (Gauss–Newton, Levenberg–Marquardt, trust-region, ...) provided by
[NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl). Afterwards it computes
parameter uncertainties from the residual Jacobian at the optimum.

While completely independent and usable on its own, users wanting the full optimal control and
experimental design suite should use [Corleone.jl](https://github.com/SciML/Corleone.jl).

## Quick start

```julia
using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve   # GaussNewton, LevenbergMarquardt, ...

# Model: du/dt = -p[1]*u
f = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
prob = ODEProblem(f, [1.0], (0.0, 5.0), [0.5])

tdata = collect(0.0:0.5:5.0)
# ... build `ydata` from a "true" trajectory plus noise ...
data = ExperimentData(tdata, ydata; σ = 0.05)

pe = ParameterEstimationLayer(prob, Tsit5(), data)
nlls = problem(pe)

sol = solve(nlls, GaussNewton())
res = estimate_parameters(pe, sol)   # point estimate + covariance + CIs
```

By default fixed parameters (e.g. via `bounds_p` with coincident lower/upper bounds on the
wrapped layer) are kept at their values and only the remaining tunable parameters are fitted.

## Relationship to Optimization.jl

CorleoneParameterEstimation primarily interfaces with **NonlinearSolve.jl**, whose
`NonlinearLeastSquaresProblem` and first-order solvers (`GaussNewton`, `LevenbergMarquardt`,
`BoundedLevenbergMarquardt`, `TrustRegion`, `Dogbox`) are the standard SciML least-squares
toolchain for this task. Optimization.jl, which Corleone uses for optimal control, does **not**
ship Gauss–Newton/Levenberg–Marquardt-style optimizers, so it is not the natural backend here.
For completeness, a thin extension (`CorleoneParameterEstimationOptimizationExtension`) exposes
an `OptimizationProblem` with the scalar `0.5 * Σ rᵢ²` objective, so scalar optimizers such as
`BFGS`/`LBFGS` can also be used if desired.
