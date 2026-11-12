# Parameter-estimation examples

These examples adapt the models and workflows contributed by **Christoph Plate
([@chplate](https://github.com/chplate))** in
[Corleone.jl PR #168](https://github.com/SciML/Corleone.jl/pull/168), at commit
`9aeeabb574d7d3fdbecfd06197ca28e4cc9dae87`, to the current annotation and experiment APIs.

| Example | Workflow |
| --- | --- |
| [Lotka–Volterra](lotka_plain/main.jl) | Fit four positive rates, report uncertainties, and use NonlinearSolve or Ipopt. |
| [Joint Lotka–Volterra experiments](lotka_multiexperiment/main.jl) | Share rates across different initial conditions and warm-start a joint fit. |
| [OED-designed controls](lotka_oed_control/main.jl) | Design a fishing input and fit with the resulting control schedule fixed. |
| [Multiple shooting](multiple_shooting/main.jl) | Fit an interior state and model rates subject to continuity equalities. |
| [SIR across two regions](sir_multicity/main.jl) | Combine per-quantity observations and estimate a local initial infection count. |
| [Robertson DAE](rober_dae/main.jl) | Fit a subset of positive rates in a stiff differential-algebraic system. |
| [Lux UDE](ude_growth/main.jl) | Train a neural reaction term, add a second experiment, and apply an L1 prior. |
| [Experiment-local annotations](experiment_layers/main.jl) | Fit a shared rate and local initial values with fixed experiment-specific controls. |

## Running

From the repository root, develop all three checked-out packages in the example
environment. This also works on Julia 1.10, which predates `[sources]` support.

```sh
julia --project=lib/CorleoneInverse/examples -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path="."), Pkg.PackageSpec(path="lib/CorleoneInverse"), Pkg.PackageSpec(path="lib/CorleoneOED")]); Pkg.instantiate()'
julia --project=lib/CorleoneInverse/examples lib/CorleoneInverse/examples/lotka_plain/main.jl
```

The scripts return CairoMakie figures and print fitted values or uncertainty
statistics. The OED and UDE examples perform several optimization runs.

## Adaptation details

Transforms and priors live on `Tunable` annotations. Observation pairs define
per-quantity grids. `Experiments` owns shared and local Lux parameters and states,
and a separate `ShootingLayer` supplies the solver. Build
`CorleoneDynamicOptProblem(shooting, experiments, constraints...; ps, st, ...)`
once, then use `NonlinearLeastSquaresProblem(model)`,
`OptimizationProblem(model, ad)`, and `ParameterEstimate(model, fit.u)`.

The multiple-shooting example enforces continuity through equality constraints.
The UDE uses a smaller 1 → 4 → 2 → 1 network for tutorial runtime and a Laplace
prior to express the original L1 penalty. OED measurement
densities rank a finite set of candidate sampling times, with six selected per
species for the synthetic experiment. Fitted trajectories use ordinary CairoMakie
lines and scatter plots.
