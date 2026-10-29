# API Reference

## Shooting layers

`ShootingLayer` returns `(trajectory, state)`. Parameter switches and scheduled
initial conditions split the solve into segments. Only active, non-fixed initial
conditions at interior segment starts contribute matching residuals. Tunable global
initial values are free; final-time assignments create no segment. See the
[shooting formulation](@ref shooting_formulation), [Parallel shooting](parallel_shooting.md)
for independent block execution and [Automatic differentiation backends](automatic_differentiation.md)
for tested gradient configurations. Segment boundaries are cached in
`st.timestops` during setup and retained in the returned state. After changing a
schedule, refresh the cache with
`st = merge(st, (; timestops = Corleone.get_timestops(layer, nothing, st)))`.

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

## Trajectories

Saved states and times are available as `trajectory.u` and `trajectory.t`, and
initial-condition matching residuals as `trajectory.residuals`. Continuous segment
handoffs are stored once; reset boundaries retain both sides, with
`trajectory.reset_indices` pointing to the outgoing states.
See the [trajectory guide](@ref trajectory_guide) for an executable reset example
and the distinction between state and parameter grids.

The trajectory retains the supplied layer, parameters, and Lux state by reference.
Parameter queries evaluate `layer.parameter` using those objects. In-place changes
to them affect subsequent parameter queries. Use
`SymbolicIndexingInterface.parameter_values_at_time(trajectory, trajectory, t)`
for values at a time, or `getp(trajectory, symbol)(trajectory)` for a parameter's
history on its parameter-change grid.

```@docs
Trajectory
Trajectory(::Any, ::Any, ::Any, ::AbstractVector, ::Any)
trajectoryplot
trajectoryplot!
```

## Symbolic bookkeeping

Retrieve the wrapper with `SymbolicIndexingInterface.symbolic_container(trajectory)`
or `SymbolicIndexingInterface.symbolic_container(layer)`. Control and quadrature
queries are defined only on `ControlSymbolCache`; original state and parameter
indices are preserved. See [Symbolic bookkeeping](symbolic_metadata.md).

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

These layers compose the public shooting layer. Their grids and activity masks
live in Lux state and can be reused between evaluations.

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

These functions return native definitions and layers. See
[Deprecated constructors](legacy_constructors.md) for the translation rules and
unsupported legacy options.

```@docs
ControlParameter
SingleShootingLayer
MultipleShootingLayer
```

## Optimization-facing contracts

See [Optimization-facing contracts](optimization_contracts.md) for structured
variable bounds, constraint ordering, and optimizer buffer requirements. The
`get_bounds` methods are documented above with symbolic bookkeeping.

```@docs
get_lower_bound
get_upper_bound
get_number_of_shooting_constraints
add_parameter_segments!
shooting_constraints
shooting_constraints!
```

## Prepared optimization facade

The core model evaluates trajectory callbacks and symbolic selectors without an
optimizer dependency. SciMLBase constructs OptimizationFunction/OptimizationProblem;
ComponentArrays supplies the default packing implementation. See
[Prepared optimization models](optimization_contracts.md#Prepared-optimization-models).

```@docs
CorleoneDynamicOptProblem
optimization_objective
optimization_constraints
optimization_constraints!
pack
default_vectorizer
default_evaluator
compile_evaluator
Corleone.SciMLBase.OptimizationFunction(::CorleoneDynamicOptProblem, ::Corleone.SciMLBase.ADTypes.AbstractADType)
Corleone.SciMLBase.OptimizationProblem(::CorleoneDynamicOptProblem, ::Corleone.SciMLBase.ADTypes.AbstractADType)
Corleone.SciMLBase.OptimizationProblem(::ShootingLayer, ::Corleone.SciMLBase.ADTypes.AbstractADType)
```
