# Corleone.jl

Corleone builds differentiable shooting layers for dynamic optimization with
[SciML](https://sciml.ai/) differential equation solvers and Lux parameter/state
containers. Parameter switching times and per-variable initial-condition resets
define the solve segments. Each evaluation returns a trajectory and updated Lux
state, with matching residuals available for continuity constraints.

## Installation

```julia
using Pkg
Pkg.add("Corleone")
```

## Getting started

Start with the [linear quadratic regulator](examples/the_linear_quadratic_regulator.md)
or the [Lotka–Volterra fishing problem](examples/the_lotka_volterra_fishing_problem.md).
Both tutorials construct a `ShootingLayer`, define an optimization objective, and
plot the resulting states and controls.

For symbolic models, follow the
[ModelingToolkit Lotka–Volterra tutorial](examples/modelingtoolkit_integration.md).

The manual also describes [parallel shooting](parallel_shooting.md), tested
[automatic differentiation backends](automatic_differentiation.md), and
[symbolic bookkeeping](symbolic_metadata.md). Constructor and field documentation
is collected in the [API reference](api.md).

This manual covers the root Corleone package. Sibling packages under `lib` and
their tutorials are outside its documentation build.
