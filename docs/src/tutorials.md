# Tutorials

These examples use Corleone's shooting layers and the CorleoneOED experimental-design
wrapper. All seven tutorials are executed during the documentation build.

- [Linear quadratic regulator](examples/the_linear_quadratic_regulator.md):
  parameter schedules, bounded optimization, and state/control plots.
- [Lotka–Volterra fishing](examples/the_lotka_volterra_fishing_problem.md):
  single shooting, lifted initial conditions, continuity constraints, and threaded
  multiple shooting.
- [ModelingToolkit integration](examples/modelingtoolkit_integration.md):
  symbolic dynamics and integral costs, metadata bounds, and single and multiple
  shooting with fixed global initial values.
- [LTI control with unknown coefficients](examples/lti_control_with_unknown_coefficients.md):
  three terminal state equalities, a scalar input, three unknown constant
  coefficients, and shared-grid multiple shooting.

## Optimal experimental design

These tutorials execute the same authoritative sources as the standalone
CorleoneOED manual. See the [OED overview](@ref oed_overview) for measurement grids,
parameter directions, optimization, and numerical scope.

- [Continuous Lotka–Volterra design](@ref lotka_oed): jointly optimize fishing and
  continuous measurement densities.
- [Discrete compartmental design](@ref compartmental_oed): allocate relaxed weights
  to candidate sampling events and construct a binary-weight problem description.
- [Multiple experiments](@ref lotka_multiexp): combine independent design trees in
  common physical coordinates and account for a global prior once.
