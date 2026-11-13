# [API reference](@id core_api)

For workflows, use the [Manual](shooting.md). This page includes docstrings for
exported APIs and supporting layer types.

## Shooting layers

```@docs
ShootingLayer
ShootingLayer(::Corleone.SciMLBase.AbstractSciMLProblem, ::Corleone.SciMLBase.AbstractSciMLAlgorithm, ::Vararg{Any})
InitialConditions
InitialConditions(::Corleone.SciMLBase.AbstractSciMLProblem, ::Vararg{Any})
Parameter
Parameter(::Corleone.SciMLBase.AbstractSciMLProblem, ::Vararg{Any})
Tunable
Constant
Annotations
```

Annotations accept a `transform` (load TransformVariables) and a univariate `prior`
(load Distributions). Transformed values are stored in internal coordinates in the
Lux parameters; layer outputs and defaults stay natural. Without an explicit
transform, `BoundedTransform` stores the bounds and uses identity coordinates.
Explicit TransformVariables transforms replace it and use unbounded optimizer
coordinates; bound keywords do not add constraints to those transforms.

```@docs
BoundedTransform
log_prior
Corleone.parameter_collections
```

See [Initializing shooting variables](initialization.md) for scalar, vector, and
count/timepoint callback defaults and helpers for existing parameter buffers.

```@docs
forward_initialization
linear_initialization
random_initialization
constant_initialization
custom_initialization
hybrid_initialization
```

## Multi-stage transitions and linkages

Chain multiple shooting stages with `Stages` and connect consecutive stages with
`StageTransition` (substitution) or `StageLinkage` (a free interface state coupled by
a residual constraint). Each layer is stored under the key `stage_i` following the
layer order. See the [multi-stage guide](@ref multistage_guide) and the
[two-stage reactor tutorial](@ref two_stage_reactor).

```@docs
StageTransition
StageLinkage
StageLinkage(::AbstractVector)
StageLinkage(::AbstractMatrix, ::AbstractMatrix, ::AbstractVector)
linkage_residual
get_number_of_linkage_constraints
Corleone.linkage_constraint_bounds
Stages
```

## Trajectories

```@docs
Trajectory
Trajectory(::Any, ::Any, ::Any, ::AbstractVector, ::Any)
trajectoryplot
trajectoryplot!
```

## Experiments

Observations and experiments belong to Corleone and can be used independently of
CorleoneInverse. Prepare the observation grid with `experiment_state`, predict with
`layer(experiment, ps, st)`, and compare predictions with `residual`.

```@docs
Observation
Experiment
experiment_state
Experiments
append_experiment
residual
n_observations
```

## Symbolic bookkeeping

```@docs
ControlSymbolCache
ControlSymbolCache(::Any)
control_symbols
control_indices
quadrature_symbols
quadrature_indices
default_bounds
Corleone.get_bounds
Corleone.is_integer_variable
```

## Internal layers and scheduling

```@docs
Corleone.PassThrough
Corleone.InitialCondition
Corleone.InitialCondition(::AbstractVector{<:Real}, ::Any)
Corleone.PiecewiseConstant
Corleone.PiecewiseConstant(::AbstractVector{<:Number}, ::Any)
Corleone.ParameterCollection
Corleone.FixedLayer
Corleone.LookUp
Corleone.lookup!
Corleone.get_timestops
Corleone.update_active_ics!
Corleone.terminal_state
Corleone.AbstractAnnotation
Corleone.DynamicVariable
Corleone.update_annotation
```

## Deprecated constructor functions

See [migration guidance](legacy_constructors.md) for legacy constructor translations.

```@docs
ControlParameter
SingleShootingLayer
MultipleShootingLayer
```

## Optimization-facing contracts

```@docs
get_lower_bound
get_upper_bound
get_integrality
get_number_of_shooting_constraints
add_parameter_segments!
adapt_timegrid!
shooting_constraints
shooting_constraints!
```

## Prepared optimization facade

```@docs
CorleoneDynamicOptProblem
EvaluationLayer
optimization_objective
optimization_constraints
optimization_constraints!
constraint_jacobian_sparsity
lagrangian_hessian_sparsity
Corleone.elementwise_series_expression
pack
default_vectorizer
default_evaluator
compile_evaluator
Corleone.SciMLBase.OptimizationFunction(::CorleoneDynamicOptProblem, ::Corleone.SciMLBase.ADTypes.AbstractADType)
Corleone.SciMLBase.OptimizationProblem(::CorleoneDynamicOptProblem, ::Corleone.SciMLBase.ADTypes.AbstractADType)
Corleone.SciMLBase.OptimizationProblem(::ShootingLayer, ::Corleone.SciMLBase.ADTypes.AbstractADType)
```

## Sublibrary integration hooks

These public, unexported functions let packages such as CorleoneInverse wrap
shooting layers in their own prediction layers. They compose evaluation
parameters, select experiments, compile observation expressions, and build
structural derivative patterns without relying on internal functions.

```@docs
Corleone.evaluation_parameters
Corleone.compose_evaluation
Corleone.compile_observation_expression
Corleone.experiment_components
Corleone.experiment_trajectories
Corleone.optimization_dependency_rows
Corleone.dependency_column_ids
Corleone.shooting_dependency_context
Corleone.trajectory_constraint_dependencies
Corleone.expression_signals
Corleone.dependency_pattern
```
