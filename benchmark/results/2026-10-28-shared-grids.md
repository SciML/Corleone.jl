# Shared-grid scalar collections — 28 October 2026

The previous runtime-buffer experiment is preserved in git stash
`WIP grouped annotation runtime buffers and permutation experiments`. This result
compares the new shared-lookup design with commit `c81906b`, using the same
`benchmark/shared_grids.jl` script and dependency environment. No sublibraries
were changed or tested.

The workload has 24 states and 24 parameters, 19 common switching/reset times,
and 20 serial integration segments. Every variable retains a scalar-valued
parameter vector. The new implementation shares one lookup per grid, with
individual ParameterCollection/FixedLayer children. The baseline uses one lookup
per variable. Both solve the same constant-derivative ODE and assert the terminal
state; packed and native trajectories are also compared.

Julia 1.12.7, AMD Ryzen 7 8700F, two Julia threads, one BLAS thread. Medians use
1,000 samples for layer applications and 300 for complete forward calls,
`evals=1`. Construction, packing, package loading, and compilation are excluded.
Each checkout runs in a separate process without concurrent test jobs.

| Operation | Baseline | Shared lookup | Speedup | Baseline bytes / allocations | Shared bytes / allocations |
|---|---:|---:|---:|---:|---:|
| Initial conditions | 1.930 µs | 0.240 µs | 8.04× | 768 / 7 | 1,024 / 9 |
| Parameters | 0.270 µs | 0.200 µs | 1.35× | 256 / 2 | 512 / 4 |
| Forward, native Lux parameters | 116.294 µs | 65.944 µs | 1.76× | 588,192 / 4,733 | 540,968 / 4,664 |
| Forward, ComponentArray parameters | 144.644 µs | 67.309 µs | 2.15× | 588,192 / 4,733 | 540,968 / 4,664 |

Complete forward calls allocate about 8.0% fewer bytes. Standalone applications
allocate an additional group-output vector; these measurements favor many
variables sharing a grid and a cheap right-hand side. They are not a general
solver speedup or a measurement of compilation latency.

Validation:

- The full Core dispatcher passed 1,421 checks; the subsequently added mixed
  scalar-type/inference test passed four more checks. The focused shared-grid
  file passes 44 checks, including the 40 already counted in Core.
- Examples passed 146 checks (50 native checks, then 96 MTK checks after updating
  one assertion for the new grouped layout). The prepared local environment was
  used to avoid resolving/downloading new dependencies.
- Plotting passed 55 checks, including fixed/tunable members sharing a reset grid.
- Documentation passed doctests, references, and HTML rendering, reusing existing
  generated tutorial pages.
- Strict Corleone precompilation passed.

The new cases cover equal grids from independent annotations, noncontiguous
indices, mixed fixed/tunable members, scalar-only buffers, canonical output and
constraint ordering, fixed-buffer expansion at inserted boundaries, initialization,
threaded forwards/ForwardDiff, ReverseDiff full pass-through solves, and Zygote
packing gradients. Existing serial/threaded AD regressions remain in Core.
