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

## Fixed controls (from optimal experimental design)

Experiments may prescribe **fixed piecewise-constant controls** as part of their design
(e.g. produced by optimal experimental design with Corleone). Each experiment supplies its
controls as a tuple of `control_index => ControlParameter` pairs (the same form as the
`controls` keyword of `SingleShootingLayer`); an experiment with no controls is left out by
passing `controls = nothing`.

```julia
# a controlled experiment: the (fixed) design applies a time-varying fishing rate
# on parameter 1, driving the model through index 1 rather than a tunable p[1]
data = ExperimentData(Experiment(t, y; σ = 0.05, u0 = [1.0],
    controls = (1 => ControlParameter(cgrid, controls = designed_control),)))

pe  = ParameterEstimationLayer(prob, Tsit5(), data)
sol = solve(problem(pe), GaussNewton())
est = estimate_parameters(pe, sol)
```

Control values are **design inputs, never optimization variables**: they are read from each
`ControlParameter` and held fixed during the fit, while the (non-control) model parameters
are estimated. A control *replaces* its parameter, so a controlled parameter index is not
estimated. Experiments may be **mixed**: some with controls, some without (`controls =
nothing`). When the model has control slots, an experiment without explicit controls is
simulated with those slots held constant at their `prob.p` default value. All experiments in
a single fit must control the same parameter indices. See
`examples/lotka_oed_control/main.jl` for an end-to-end workflow that solves an
optimal experimental design problem with CorleoneOED and then estimates the parameters with
the designed control fixed, and `examples/multiple_shooting/main.jl` for the
multiple-shooting variant of that fit.

## Per-quantity observation times

Different measured quantities of a model may be sampled at **different time points**. This
is the second `Experiment` constructor form, taking per-quantity time vectors and value
vectors:

```julia
data = ExperimentData(Experiment(
    [prey_obs_times, predator_obs_times],   # each quantity has its own observation times
    [prey_data, predator_data],             # values at those times
; σ = 0.05, u0 = [1.0, ...]))
```

The `k`-th entry of the two vectors refers to the `k`-th detected quantity of the layer's
`observed` map, which is measured only at its own time points. This is in contrast to the
shared-grid matrix form `Experiment(t, y)` (an `n_obs × length(t)` matrix in which every
quantity is measured at every time point). The two forms can be freely mixed inside a single
`ExperimentData`. This is particularly useful in an optimal-experimental-design loop, where
each observable species is sampled at the (generally different) times that the design
selects (see the example). Use the accessors `per_quantity`, `observation_times`,
`observation_times_per_quantity` and `n_observations` to introspect the data.

## Relationship to Optimization.jl

CorleoneParameterEstimation primarily interfaces with **NonlinearSolve.jl**, whose
`NonlinearLeastSquaresProblem` and first-order solvers (`GaussNewton`, `LevenbergMarquardt`,
`BoundedLevenbergMarquardt`, `TrustRegion`, `Dogbox`) are the standard SciML least-squares
toolchain for this task. Optimization.jl, which Corleone uses for optimal control, does **not**
ship Gauss–Newton/Levenberg–Marquardt-style optimizers, so it is not the natural backend here.
For completeness, a thin extension (`CorleoneParameterEstimationOptimizationExtension`) exposes
an `OptimizationProblem` with the scalar `0.5 * Σ rᵢ²` objective, so scalar optimizers such as
`BFGS`/`LBFGS` -- or any MathOptInterface (MOI) optimizer such as `UnoSolver.Optimizer(preset =
"filtersqp")` via `OptimizationMOI` -- can also be used if desired. The extension is opt-in and
activates once both `Optimization` and `ComponentArrays` are loaded; see the final section of
`examples/lotka_plain/main.jl` for a worked example solving the same Lotka-Volterra fit with
`UnoSolver`.
