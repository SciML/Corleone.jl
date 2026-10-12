# API Reference

## Corleone

```@docs
Corleone.Trajectory
Corleone.ControlParameter
Corleone.SingleShootingLayer
Corleone.MultipleShootingLayer
Corleone.default_initialization
Corleone.random_initialization
Corleone.forward_initialization
Corleone.linear_initialization
Corleone.custom_initialization
Corleone.constant_initialization
Corleone.hybrid_initialization
Corleone.CorleoneDynamicOptProblem
```

### Developer interface

These hooks are the contract used by CorleoneOED and by extensions that add new
shooting-layer wrappers. They are documented for package developers; ordinary users
should prefer the exported constructors and layer methods above.

```@docs
Corleone.get_block_structure
Corleone.get_bounds
Corleone.get_controls
Corleone.get_number_of_shooting_constraints
Corleone.get_problem
Corleone.get_timegrid
Corleone.get_tspan
Corleone.get_tunable
Corleone.shooting_constraints
Corleone.shooting_constraints!
```

## CorleoneOED

```@docs
CorleoneOED.OEDLayer
CorleoneOED.fisher_information
CorleoneOED.observed_equations
CorleoneOED.sensitivities
CorleoneOED.local_information_gain
CorleoneOED.global_information_gain
CorleoneOED.MultiExperimentLayer
CorleoneOED.AbstractCriterion
CorleoneOED.ACriterion
CorleoneOED.DCriterion
CorleoneOED.ECriterion
CorleoneOED.FisherACriterion
CorleoneOED.FisherECriterion
CorleoneOED.FisherDCriterion
```

## CorleoneParameterEstimation

### Data

```@docs
CorleoneParameterEstimation.Experiment
CorleoneParameterEstimation.ExperimentData
CorleoneParameterEstimation.experiments
CorleoneParameterEstimation.n_experiments
CorleoneParameterEstimation.n_observations
CorleoneParameterEstimation.observation_times
CorleoneParameterEstimation.observation_times_per_quantity
CorleoneParameterEstimation.per_quantity
CorleoneParameterEstimation.initial_conditions
CorleoneParameterEstimation.controls
CorleoneParameterEstimation.has_controls
```

### Parameter-estimation layer and problem

```@docs
CorleoneParameterEstimation.ParameterEstimationLayer
CorleoneParameterEstimation.CorleoneParameterEstimationProblem
CorleoneParameterEstimation.problem
CorleoneParameterEstimation.residuals!
CorleoneParameterEstimation.residual
CorleoneParameterEstimation.model_values
CorleoneParameterEstimation.n_shooting_constraints
```

### Parameter transforms

```@docs
CorleoneParameterEstimation.AbstractParameterTransform
CorleoneParameterEstimation.Identity
CorleoneParameterEstimation.Affine
CorleoneParameterEstimation.Log
CorleoneParameterEstimation.Logit
CorleoneParameterEstimation.to_internal
CorleoneParameterEstimation.from_internal
CorleoneParameterEstimation.dθ_dφ
```

### Estimates, uncertainties and diagnostics

```@docs
CorleoneParameterEstimation.ParameterEstimate
CorleoneParameterEstimation.ParameterEstimationResult
CorleoneParameterEstimation.estimate_parameters
CorleoneParameterEstimation.fitted_parameters
CorleoneParameterEstimation.covariance
CorleoneParameterEstimation.standard_errors
CorleoneParameterEstimation.confidence_interval
CorleoneParameterEstimation.r_squared
CorleoneParameterEstimation.adjusted_r_squared
CorleoneParameterEstimation.correlation
CorleoneParameterEstimation.condition_number
```

### Parameter regularization

```@docs
CorleoneParameterEstimation.AbstractRegularizer
CorleoneParameterEstimation.L2Regularizer
CorleoneParameterEstimation.L1Regularizer
CorleoneParameterEstimation.ElasticNetRegularizer
CorleoneParameterEstimation.n_regularization_residuals
```

### Profile likelihood

```@docs
CorleoneParameterEstimation.profile_likelihood
CorleoneParameterEstimation.ProfileCurve
CorleoneParameterEstimation.ProfileLikelihoodEstimate
CorleoneParameterEstimation.profiles
```

## OptimalControlBenchmarks

```@docs
OptimalControlBenchmarks.load_benchmarks
OptimalControlBenchmarks.run_all
OptimalControlBenchmarks.OptimalControlBenchmark
OptimalControlBenchmarks.BenchmarkGrids
```
