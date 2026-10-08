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

## CorleoneBase

See [Sequential problems](@ref corleonebase) for callback conventions, solver
keywords, stepping, and failure handling, and [manual single shooting](@ref base_fishing)
for explicit objective and bound construction without shooting layers.

See [CorleoneBase API](@ref base_api).

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

## OptimalControlBenchmarks

```@docs
OptimalControlBenchmarks.load_benchmarks
OptimalControlBenchmarks.run_all
OptimalControlBenchmarks.OptimalControlBenchmark
OptimalControlBenchmarks.BenchmarkGrids
```
