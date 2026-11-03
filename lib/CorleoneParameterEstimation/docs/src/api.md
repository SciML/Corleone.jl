# [API reference](@id pe_api)

The entries below document CorleoneParameterEstimation-owned APIs and methods of shared
functions. Re-exported Corleone and NonlinearSolve APIs retain their own documentation in
the [Corleone manual](https://sciml.github.io/Corleone.jl/dev/). The shared optimization
helpers include `constraint_jacobian_sparsity` and `lagrangian_hessian_sparsity` for
supported shooting problems.

## Data and experiments

```@docs
Experiment
ExperimentData
experiments
n_experiments
n_observations
observation_times
observation_times_per_quantity
per_quantity
initial_conditions
controls
has_controls
```

## Layers and problems

```@docs
ParameterEstimationLayer
CorleoneParameterEstimationProblem
problem
model_values
residual
residuals!
n_shooting_constraints
```

## Fitting and uncertainty

```@docs
ParameterEstimate
ParameterEstimationResult
estimate_parameters
fitted_parameters
covariance
standard_errors
confidence_interval
r_squared
adjusted_r_squared
correlation
condition_number
```

## Parameter transforms

```@docs
AbstractParameterTransform
Identity
Affine
Log
Logit
```

## Regularization

```@docs
AbstractRegularizer
L2Regularizer
L1Regularizer
ElasticNetRegularizer
CorleoneParameterEstimation.n_regularization_residuals
```

## Profile likelihood

```@docs
ProfileCurve
ProfileLikelihoodEstimate
profile_likelihood
profiles
```
