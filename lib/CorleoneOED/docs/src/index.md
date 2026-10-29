# CorleoneOED

CorleoneOED augments a differential equation with parameter sensitivities and
Fisher information to design controls and measurement schedules. It uses the
rewritten Corleone shooting, bounds, and optimization contracts.

The [package README](https://github.com/SciML/Corleone.jl/tree/main/lib/CorleoneOED)
explains the formulation, measurement grids, optimization interface, and numerical
scope. The package's `examples/` directory contains executable Julia tutorials
for continuous Lotka–Volterra designs, discrete compartmental measurements, and
multiple experiments.

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

## Attribution

The measurement, multi-experiment, and DAE work is adapted from
[Christoph Plate (@chplate)'s contributions](https://github.com/AlCap23/Corleone.jl/commit/7443580115fa432e4996a67b45a9028b3d10e7ed)
on the fork's `main` branch. The original MIT license and authorship are retained.
