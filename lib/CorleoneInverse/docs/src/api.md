# [API reference](@id inverse_api)

Observations, experiments, predictions, and residuals are documented in the
[core API reference](@ref core_api).

```@docs
ParameterEstimate
mean_squared_residual
residual_jacobian_sparsity
CorleoneDynamicOptProblem(::ShootingLayer, ::CorleoneInverse.Experiments, ::Vararg{Any})
CorleoneInverse.SciMLBase.NonlinearLeastSquaresProblem(::CorleoneInverse.ExperimentDynamicOptProblem)
CorleoneInverse.SciMLBase.NonlinearLeastSquaresProblem(::ShootingLayer, ::CorleoneInverse.Experiments, ::Any, ::Any)
CorleoneInverse.SciMLBase.OptimizationProblem(::ShootingLayer, ::CorleoneInverse.Experiments, ::Any, ::Any, ::CorleoneInverse.AbstractADType)
```
