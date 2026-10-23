# API Reference

## Corleone

`ShootingLayer` (including the legacy `SingleShootingLayer` constructor) returns
`(trajectory, state)`. Saved states and times are available as `trajectory.u` and
`trajectory.t`, and initial-condition matching residuals as `trajectory.residuals`.
Continuous segment handoffs are stored once; reset boundaries retain both sides,
with `trajectory.reset_indices` pointing to the outgoing states.

The trajectory retains the supplied `layer`, `ps`, and `st` directly. Parameter
queries evaluate `layer.parameter` using those objects, preserving differentiation
through the parameters. In-place changes to them affect subsequent parameter queries.
`SymbolicIndexingInterface.parameter_values_at_time(trajectory, trajectory, t)`
evaluates parameters at a time, and `getp(trajectory, symbol)(trajectory)` retrieves
a parameter's history on the combined parameter-change grid. No parameter-history
object is stored in the trajectory.

For tested backend configurations, solver settings, limitations, and validation
results, see [Automatic differentiation backends](automatic_differentiation.md).

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

### SymbolicUtils-owned names re-exported via Symbolics

`CorleoneOED` builds its facade with `@reexport using Symbolics`. The following
names are owned by SymbolicUtils and only reach CorleoneOED because Symbolics
re-exports them; CorleoneOED does not define them.

- [`SymbolicUtils.Unknown`](@extref)
- [`SymbolicUtils.shape`](@extref)
- `scalarize`
- `unwrap`

## OptimalControlBenchmarks

```@docs
OptimalControlBenchmarks.load_benchmarks
OptimalControlBenchmarks.run_all
OptimalControlBenchmarks.OptimalControlBenchmark
OptimalControlBenchmarks.BenchmarkGrids
```
