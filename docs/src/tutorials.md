# Tutorials

Learn to control a dynamical system, build constrained models from equations, and
choose informative measurements. Each tutorial is an executed Pluto notebook with
plots and a source download. Start with the fundamentals; the later sections build
on the same shooting-layer and optimization workflow.

## Control fundamentals

Basic Julia and ordinary differential equations are enough to start.

1. [Track a target with linear quadratic control](examples/optimal_control/linear_quadratic.md).
   Build a shooting layer, initialize its parameters, optimize a piecewise constant
   input, and inspect state and control trajectories.
2. [Control predator–prey dynamics](examples/optimal_control/lotka_fishing.md).
   Add bounded fishing controls, lifted initial conditions, continuity constraints,
   and threaded multiple shooting.

## Symbolic models and constraints

Continue here after the fundamentals to use the ModelingToolkit frontend.

1. [Build constrained control models from equations](examples/modelingtoolkit/lotka.md).
   Express symbolic dynamics and integral costs, set metadata bounds, and compare
   single and multiple shooting with fixed global initial values.
2. [Estimate coefficients while controlling a system](examples/modelingtoolkit/lti.md).
   Optimize a scalar input and three unknown constant coefficients while satisfying
   terminal state equalities with shared-grid multiple shooting.

## Measurement design

Use CorleoneOED to choose when to observe a system and how to combine experiments.
The [OED overview](@ref oed_overview) introduces measurement grids, parameter
sensitivities, and information criteria.

1. [Optimize continuous measurement schedules](oed_examples/experimental_design/lotka_oed.md).
   Jointly optimize fishing and continuous measurement densities.
2. [Choose discrete sampling times](oed_examples/experimental_design/compartmental_oed.md).
   Allocate relaxed weights to candidate samples and construct a binary-weight
   problem description.
3. [Combine information across experiments](oed_examples/experimental_design/lotka_multiexperiments.md).
   Combine independent design trees in common physical coordinates and account for
   a global prior once.
