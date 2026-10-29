# Shared-grid AD audit — 28 October 2026

Root Corleone only, Julia 1.12.7. Production source was unchanged by this audit.

The serial cases use two tunable controls and one fixed control on a common grid.
The joint case also groups three tunable shooting states and adds parameter
segments at full resets. Input dimensions are 16 (controls) and 26 (joint).
Gradient checks use two inputs, central differences with h=1e-5 and h=5e-6,
reused DifferentiationInterface preparation, and unchanged-input assertions.

| Backend | Controls | Joint states/parameters | Maximum absolute gradient error |
|---|---|---|---:|
| ForwardDiff | Pass | Pass | 3.83e-10 |
| ReverseDiff pass-through | Pass | Pass | 3.89e-10 |
| Zygote default sensitivities | Pass | Pass | 3.86e-10 |
| Mooncake reverse pass-through | Pass | Pass | 3.79e-10 |
| FiniteDiff reference | Pass | Pass | 4.95e-10 |
| ReverseDiff default sensitivities | Existing failure | Existing failure | — |
| Enzyme forward | Existing failure | Existing failure | — |

Pass-through cases use an in-place `FullSpecialize` ODEFunction and
`SensitivityADPassThrough()`. ReverseDiff's default path still supplies a tracked
matrix to a segment handoff expecting solution metadata. Enzyme forward still
attempts a duplicated ShootingLayer annotation rejected by Lux, despite a Const
loss-function annotation and runtime activity. Neither is a new shared-grid
failure.

Maintained regression tests:

- Differentiation: 70 passed, including 40 new shared-grid analytic checks.
  ForwardDiff and Zygote cover serial/threaded layers; ReverseDiff covers serial
  pass-through solves. Reverse block evaluation for Zygote remains serial.
- Distributed: 25 passed with two local workers, including the new shared-grid
  analytic fixture for ForwardDiff.
- The new fixture differentiates two shared tunable parameters, shooting values,
  a fixed reset value and a fixed parameter stored in Lux state, at two inputs.
  Values and gradients are checked against a closed-form solution.

Serial audit reproduction:

    CORLEONE_AD_SHARED_GRIDS=true julia --project=<AD-env> benchmark/shooting_ad.jl <backend>

The audit reduced benchmark samples to one and ran independent backend processes
concurrently. Recorded times are deliberately omitted: this is a correctness and
compatibility check, not an AD performance comparison. Mooncake/ReverseDiff
parallel execution and Enzyme reverse were not established by these checks.
