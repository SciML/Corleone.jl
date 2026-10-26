# API Reference

## Shooting layers

`ShootingLayer` returns `(trajectory, state)`. Parameter switches and scheduled
initial conditions split the solve into segments. Only active, non-fixed initial
conditions contribute matching residuals. See [Parallel shooting](parallel_shooting.md)
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
```

See [Initializing shooting variables](initialization.md) for scalar, vector, and
count/timepoint callback defaults.

## Trajectories

Saved states and times are available as `trajectory.u` and `trajectory.t`, and
initial-condition matching residuals as `trajectory.residuals`. Continuous segment
handoffs are stored once; reset boundaries retain both sides, with
`trajectory.reset_indices` pointing to the outgoing states.

The trajectory retains the supplied layer, parameters, and Lux state by reference.
Parameter queries evaluate `layer.parameter` using those objects. In-place changes
to them affect subsequent parameter queries. Use
`SymbolicIndexingInterface.parameter_values_at_time(trajectory, trajectory, t)`
for values at a time, or `getp(trajectory, symbol)(trajectory)` for a parameter's
history on its parameter-change grid.

```@docs
Trajectory
Trajectory(::Any, ::Any, ::Any, ::AbstractVector, ::Any)
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
