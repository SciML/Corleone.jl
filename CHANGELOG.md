# Rewrite release

Corleone and CorleoneOED contain breaking public API changes relative to their
preceding releases. OptimalControlBenchmarks and CorleoneGame are prepared for
their initial releases. Test, documentation, and sibling-package environments
target the new release series together.

## Corleone 2.0.0

- Replace the former shooting types with annotation-based `ShootingLayer`
  construction. Deprecated constructor functions return the new layer type.
- Return `(trajectory, state)` from layer evaluation, with matching residuals
  stored in the trajectory and a new Lux parameter/state layout.
- Rework optimization construction around `CorleoneDynamicOptProblem`, with
  keyword-selected vectorization and evaluation extensions.
- Add shared-grid annotations, threaded/distributed shooting, and structural
  sparse Jacobian and exact Hessian support.

## CorleoneOED 2.0.0

- Adapt experimental-design layers, measurements, and multi-experiment handling
  to the Corleone 2 parameter, state, and trajectory interfaces.
- Replace the separate Optimization extension with the shared Corleone
  optimization bridge.

## OptimalControlBenchmarks 0.1.0 (initial release)

- Provide a collection of optimal-control benchmark problems with a shared
  registry, grid handling, and Corleone 2 solver integration.
- Include examples and documentation integrated into the Corleone manual.

## CorleoneGame 0.1.0 (initial release)

- Provide open-loop N-player games using Corleone 2, with player-specific control
  and parameter ownership, single and multiple shooting, warm starts, and audits.
- Default both shooting modes to Ipopt, with UnoSolver also available.
