# API Reference

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
layer order. See the [two-stage reactor tutorial](examples/optimal_control_of_a_two_stage_reactor_system.md).

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
get_number_of_shooting_constraints
add_parameter_segments!
shooting_constraints
shooting_constraints!
```

## Prepared optimization facade

```@docs
CorleoneDynamicOptProblem
optimization_objective
optimization_constraints
optimization_constraints!
constraint_jacobian_sparsity
lagrangian_hessian_sparsity
pack
default_vectorizer
default_evaluator
compile_evaluator
Corleone.SciMLBase.OptimizationFunction(::CorleoneDynamicOptProblem, ::Corleone.SciMLBase.ADTypes.AbstractADType)
Corleone.SciMLBase.OptimizationProblem(::CorleoneDynamicOptProblem, ::Corleone.SciMLBase.ADTypes.AbstractADType)
Corleone.SciMLBase.OptimizationProblem(::ShootingLayer, ::Corleone.SciMLBase.ADTypes.AbstractADType)
```
