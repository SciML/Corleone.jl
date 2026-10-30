# [API reference](@id oed_api)

The entries below document OED-owned APIs and OED-specific methods of shared
functions. Re-exported Corleone and Symbolics APIs retain their own documentation.
See the [design guide](@ref oed_designs) for links to the shared Corleone manual.

## Measurements and layers

```@docs
AbstractMeasurement
ContinuousMeasurement
DiscreteMeasurement
OEDLayer
MultiExperimentLayer
```

## Information and sampling

```@docs
fisher_information
sensitivities
observed_equations
sampling_sums
sampling_sums!
continuous_sampling_sums
discrete_sampling_sums
local_information_gain
global_information_gain
update_fim
```

## Optimization facade

```@docs
Corleone.CorleoneDynamicOptProblem(::Union{OEDLayer,MultiExperimentLayer})
CorleoneOED.SciMLBase.OptimizationProblem(::Union{OEDLayer,MultiExperimentLayer}, ::AbstractCriterion)
```

## Criteria

```@docs
AbstractCriterion
ACriterion
DCriterion
ECriterion
FisherACriterion
FisherDCriterion
FisherECriterion
```
