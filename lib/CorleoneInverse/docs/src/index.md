# [Parameter estimation](@id inverse_index)

CorleoneInverse fits the tunable values of a Corleone `ShootingLayer` to
measurements. You describe the data as an [`Experiment`](@ref), declare which
parameters and initial states to estimate with the usual `Tunable` and `Constant`
annotations, and solve with NonlinearSolve.jl or Optimization.jl. A fit can then
report covariances and confidence intervals.

## A first fit

We estimate the decay rate of ``\dot u = -p u`` from ten noise-free samples of
``e^{-1.3 t}``, starting from the guess ``p = 0.5``:

```@example first_fit
using CorleoneInverse, OrdinaryDiffEqTsit5, NonlinearSolve

ode = ODEProblem((u, p, t) -> -p[1] .* u, [1.0], (0.0, 2.0), [0.5])
shooting = ShootingLayer(ode, Tsit5(), Tunable(1) => nothing)
t = collect(0.2:0.2:2.0)
experiment = Experiment(1 => (t, exp.(-1.3 .* t), 0.01))
model = CorleoneDynamicOptProblem(shooting, experiment)
fit = solve(NonlinearLeastSquaresProblem(model), LevenbergMarquardt())
ParameterEstimate(model, fit.u)
```

`Experiment(1 => (t, y, σ))` says that state 1 was measured at times `t` with values
`y` and noise standard deviation `σ`. [`CorleoneDynamicOptProblem`](@ref) sets up
`ps` and `st` itself; pass both explicitly to reuse an existing Lux setup.

## Residuals and objectives

For every observed value the model forms the noise-scaled residual

```math
r = \frac{y - h(u(t), p, t)}{\sigma},
```

where ``h`` is the observed quantity. Give a noise scale for every observation or
for none; without scales, residuals are unweighted.

Two solver problems are available:

| Solver problem | Use it for |
| --- | --- |
| `NonlinearLeastSquaresProblem(model)` | Plain least squares with annotation bounds and untransformed Normal priors. |
| `OptimizationProblem(model, ad)` | General priors, user constraints, and multiple-shooting equalities. |

The scalar objective used by `OptimizationProblem` is the mean squared residual
minus `log_prior`. Pass `reducer = x -> sum(x) / 2` to the model constructor for
the Gaussian half-sum, which is the scaling the [statistics](#Uncertainty-estimates)
assume when priors are present. The reducer receives the squared, noise-scaled
residuals; priors and constraints are added separately.

## Uncertainty estimates

[`ParameterEstimate`](@ref)`(model, sol.u)` evaluates the observation Jacobian ``J``
at the fit and returns the covariance

```math
\left(J^\top J / s^2 + H\right)^{-1},
```

where ``H`` is the Hessian of the negative log prior. Equality constraints,
including multiple-shooting continuity, are respected by restricting the
covariance to their null space.

- With known noise scales, ``s^2 = 1`` and intervals use normal quantiles.
- Without them, ``s^2`` is the residual variance and intervals use Student-t
  quantiles.
- Results are reported in natural coordinates: the covariance through the delta
  method, and confidence intervals by mapping their endpoints from internal
  coordinates.

!!! note "Assumptions"
    The estimate is a local, linearized approximation. It assumes an interior
    optimum, away from active inequalities and parameter bounds. With priors, it
    also assumes known noise scales and a fit of half the weighted residual sum
    of squares minus the log prior: use least squares with Normal priors, or
    `reducer = x -> sum(x) / 2` with `OptimizationProblem`.

## Profile likelihoods

Loading LikelihoodProfiler.jl enables a constructor that wraps the original
optimization problem and a fitted `ParameterEstimate`:

```julia
using Optimization, OptimizationOptimJL, LikelihoodProfiler
using Distributions: quantile, Chisq

# Known observation noise scales; a Gaussian negative log-likelihood.
model = CorleoneDynamicOptProblem(shooting, experiment; reducer = x -> sum(x) / 2)
problem = OptimizationProblem(model, AutoForwardDiff())
fit = solve(problem, LBFGS())
estimate = ParameterEstimate(model, fit.u)
profile = ProfileLikelihoodProblem(problem, estimate;
    profile_lower = -5.0, profile_upper = 5.0,
    threshold = quantile(Chisq(1), estimate.level) / 2)
profiles = solve(profile, OptimizationProfiler(optimizer = LBFGS()))
```

LikelihoodProfiler's default threshold is
the full chi-squared quantile, so halve it for a negative log-likelihood objective.
The wrapper passes the objective and threshold through unchanged; choose their
scaling consistently, especially with unweighted residuals or custom reducers.

Controls are excluded from the profile targets by default and remain free nuisance
parameters during reoptimization. Set `include_controls = true` to target all
tunable values, or use `idxs = [1, 3]` to choose packed indices explicitly,
including controls. The confidence level defaults to `estimate.level`.

The original optimization problem retains its bounds and constraints; algorithm
and optimizer support for constrained profiles follows LikelihoodProfiler.
LikelihoodProfiler 1.6.1's stepwise profiler requires a backend with an Optimization.jl
`OptimizationCache`, such as OptimizationOptimJL, rather than OptimizationMOI.
Supply finite `profile_lower` and `profile_upper` bounds if its bounds are infinite;
these bounds and the resulting profiles are in internal optimizer coordinates.
The estimate retains the fitted internal values even when `estimate.x` reports
transformed natural values. Use the same model and packing for the estimate and
the optimization problem.

## Where to go next

- [Observations and parameters](@ref inverse_observations) covers measurement
  expressions, transforms, priors, and trainable observation parameters.
- [Multiple experiments](@ref inverse_experiments) covers shared and local values,
  warm starts, constraints across experiments, and sparse derivatives.
- The [API reference](@ref inverse_api) lists all functions.

## Worked examples

- [Lotka–Volterra estimation](@ref inverse_lotka_plain): positive parameters and
  uncertainty estimates.
- [Profile likelihoods](@ref inverse_likelihood_profile): transformed parameters,
  fitted controls, and profile confidence intervals.
- [Joint Lotka–Volterra experiments](@ref inverse_lotka_multiexperiment): shared
  parameters across experiments.
- [Controls from experimental design](@ref inverse_lotka_oed_control): estimation
  with fixed, OED-designed inputs.
- [Multiple shooting](@ref inverse_multiple_shooting): continuity constraints in a fit.
- [SIR estimation across two regions](@ref inverse_sir_multicity): local initial
  states and shared rates.
- [Robertson DAE](@ref inverse_rober_dae): a stiff differential-algebraic model.
- [Neural reaction term](@ref inverse_ude_growth): a Lux network with a sparsity prior.
- [EGF signaling](@ref inverse_egf_signaling): a published signaling model across
  six doses, including an identifiability check.

Most examples are adapted from Christoph Plate
([@chplate](https://github.com/chplate))'s
[PR #168](https://github.com/SciML/Corleone.jl/pull/168).
