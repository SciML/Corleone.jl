# [Parameter estimation](@id inverse_index)

CorleoneInverse fits the tunable parameters of a Corleone `ShootingLayer` to
measurements. Corleone provides the experiment and observation types. An [`Experiment`](@ref) stores one [`Observation`](@ref) per observed
quantity as a `selector => (t, y)` or `selector => (t, y, σ)` pair, where the
selector is a state symbol or index, or a scalar map `(u, p, t) -> y`. Each quantity
has its own observation times; an experiment may also prescribe its initial state:

```julia
experiment = Experiment(:u₁ => (t₁, y₁, 0.05), :u₂ => (t₂, y₂, 0.05); u0 = [0.5, 2.0])
```

A shooting layer predicts an experiment with `layer(experiment, ps, st)`, where
[`experiment_state`](@ref) merges the observation times into the segment grid of
`st`. All experiments share the layer's parameters `ps`. [`residual`](@ref)
concatenates the observation residuals

```math
r = \frac{y - h(u(t), p, t)}{\sigma}
```

over all observations and experiments.

## Parameters, transforms, and priors

Tunable parameters, fixed parameters and bounds are declared with Corleone's
annotations on the shooting layer: `Tunable(i; kind = Val(:parameter))` is estimated,
`Constant(i; kind = Val(:parameter))` is fixed. Tunable initial conditions are
estimated as well. Annotations also accept an increasing scalar `transform` from
TransformVariables and a univariate `prior` from Distributions, both on natural
values:

```julia
Tunable(1; kind = Val(:parameter), default = 1.2, transform = asℝ₊, prior = LogNormal(0.0, 0.5))
```

The Lux parameters, and hence the optimizer, then work in internal coordinates,
while layer outputs and defaults stay natural. Without an explicit transform,
`BoundedTransform` stores identity-coordinate bounds. Explicit TransformVariables
transforms replace it and use unbounded optimizer coordinates.

## Solving

Both builders take `(layer, experiments, ps, st)`, prepare the experiment states
once, and optimize the packed internal parameters.

- `NonlinearLeastSquaresProblem(layer, experiments, ps, st)` builds a least-squares
  problem for NonlinearSolve. Normal priors on untransformed parameters add the
  residual rows `(θ - μ) / σ`. It does not support bounds, other priors, or multiple
  shooting.
- `OptimizationProblem(layer, experiments, ps, st, ad)` minimizes half the residual
  sum of squares minus `log_prior` (maximum a posteriori), with the layer's bounds and
  the multiple-shooting matching constraints. Multiple shooting is supported for a single experiment, because the
  shooting node values are part of the shared parameters.

## Statistics

[`ParameterEstimate`](@ref)`(layer, experiments, sol.u, ps, st)` evaluates the observation Jacobian `J` at the fit and
returns the covariance `(JᵀJ / s² + H)⁻¹`, where `H` is the Hessian of the negative
log prior, restricted to the null space of the matching
constraints for multiple shooting. Known noise scales use `s² = 1` and normal
quantiles; otherwise `s²` is the residual variance and intervals use Student-t
quantiles. Results are reported in natural coordinates: the covariance with the delta
method, the confidence intervals by mapping the internal endpoints.
