# Bounded identity transforms — 10 November 2026

Corleone now stores bounds inside `BoundedTransform`, which implements identity
forward/inverse maps and a zero log-Jacobian. Explicit TransformVariables maps
remain separate objects and report unbounded optimizer coordinates. The public
annotation constructors retain their keyword interface.

The baseline is a source snapshot of the working tree immediately before this
refactor, including the existing inverse-problem work. Both source trees used the
same resolved dependency versions. The source snapshots and raw measurements were
kept in `/tmp/corleone-bounds-review` during this run; no baseline commit was made.

## Method

Julia 1.12.7, AMD Ryzen 7 8700F, one Julia thread and one BLAS thread. Each source
version ran twice in fresh processes, without concurrent test or benchmark jobs.
Tables report the range of the two process medians, not confidence intervals.
Package loading, precompilation, and first-use compilation are excluded.

`benchmark/bounded_transforms.jl` uses BenchmarkTools with 500 samples and a
0.6-second sampling budget per case. It checks identity/transformed lookup outputs,
identity-buffer sharing, and the expected terminal state before measuring. The
single-state ODE is `u′ = -p*u`, with `p=2`, `u(0)=1`, and `t ∈ [0,2]`. Both
parameterizations produce the same trajectory. Captured inputs are reused during
annotation-construction microbenchmarks, so compiler constant folding is possible.

## Small workloads

| Operation | Baseline median range (µs) | Updated median range (µs) | Baseline bytes / allocations | Updated bytes / allocations |
|---|---:|---:|---:|---:|
| Identity layer construction | 13.290–13.390 | 12.729–13.155 | 12,272 / 263 | 12,288 / 249 |
| Identity Lux setup | 1.344–1.351 | 1.288–1.371 | 1,824 / 49 | 1,920 / 49 |
| Identity bounds query | 7.574–7.596 | 7.470–9.002 | 6,784 / 195 | 7,008 / 191 |
| Identity log prior | 1.021–1.048 | 1.067–1.114 | 1,488 / 29 | 1,776 / 29 |
| Identity ODE forward | 10.320–10.550 | 10.610–10.945 | 48,368 / 1466 | 48,368 / 1466 |
| Transformed layer construction | 17.190–17.235 | 16.565–16.640 | 12,272 / 263 | 11,552 / 241 |
| Transformed Lux setup | 1.945–1.976 | 1.946–1.958 | 1,888 / 51 | 1,888 / 51 |
| Transformed bounds query | 10.146–10.184 | 6.852–6.878 | 6,784 / 195 | 6,432 / 177 |
| Transformed log prior | 1.084–1.112 | 1.073–1.673 | 1,488 / 29 | 1,488 / 29 |
| Transformed ODE forward | 10.410–10.720 | 10.680–11.034 | 48,336 / 1465 | 48,336 / 1465 |

| Microbenchmark | Baseline median range (ns) | Updated median range (ns) | Baseline bytes / allocations | Updated bytes / allocations |
|---|---:|---:|---:|---:|
| Identity lookup | 1.790–1.980 | 1.980 | 0 / 0 | 0 / 0 |
| Transformed lookup | 3.670–3.690 | 3.670–3.680 | 0 / 0 | 0 / 0 |
| 100-value identity array setup | 1.789–1.789 | 1.590–1.780 | 0 / 0 | 0 / 0 |
| Scalar annotation construction | 0.990 | 0.990 | 0 / 0 | 0 / 0 |
| Two-variable identity annotation | 1.790–1.980 | 1.790–1.980 | 0 / 0 | 0 / 0 |
| Two-variable mixed annotation | 1.980 | 38.429–38.846 | 0 / 0 | 240 / 6 |

The identity layer's deep storage size increases from 2,096 to 2,112 bytes in the
single-parameter model. The transformed layer remains 2,096 bytes: no wrapper is
added to TransformVariables objects. Setup, bounds queries, and prior traversal
copy some larger identity metadata objects; their allocated bytes increase even
though allocation counts do not increase. This is not a universally zero-cost
representation change.

Mixed grouped annotations normalize each entry into its own transform object. The
two-variable construction case adds 240 bytes in six allocations and approximately
37 ns relative to the captured-input baseline. This occurs during construction,
not during lookup or differential-equation evaluation.

## Shared-grid workload

The unchanged `benchmark/shared_grids.jl` uses 24 states and 24 parameters sharing
19 switching/reset times, with 20 integration segments. It checks the terminal
state and native/ComponentArray agreement. BenchmarkTools uses 1,000 samples for
lookups and 300 for full forwards, with one evaluation per sample. These processes
were pinned to CPU 6. Run order was baseline/updated, then updated/baseline to
check order-dependent timing variation.

| Operation | Baseline median range (µs) | Updated median range (µs) | Bytes / allocations, both versions |
|---|---:|---:|---:|
| Initial-condition lookup | 0.250–0.260 | 0.230–0.290 | 1,024 / 9 |
| Parameter lookup | 0.190–0.210 | 0.190–0.200 | 512 / 4 |
| Native forward | 63.614–67.125 | 67.125–67.724 | 540,968 / 4664 |
| ComponentArray forward | 69.625–72.260 | 71.710–73.484 | 540,968 / 4664 |

Lookup costs remain comparable, and forward allocation totals are unchanged.
Native forwards were 0.9–5.5% slower in the paired runs; ComponentArray forwards were
1.7–3.0% slower. The timing ranges overlap or nearly overlap, so these runs do not
establish a substantial runtime regression, but they also do not establish exactly
zero overhead. The small ODE's identity forwards were 0.6–6.1% slower, with identical
48,368-byte / 1,466-allocation totals. These are warmed local measurements, not
cold-start or universal performance guarantees.

## Validation

- Final focused root run: 606 checks passed across annotations, transforms/priors,
  shared-grid AD, symbolic metadata, optimization contracts, shooting initialization,
  and parameter-segment insertion. Annotation initialization passed another 126
  checks in the earlier focused run.
- CorleoneInverse Core: 50 checks passed, covering regression, priors, transformed
  fitting, and multiple shooting.
- Root QA: 24 passed and one pre-existing expected-broken import check.
- CorleoneInverse QA: 20 passed.
- `git diff --check` passed.

The transform tests cover stored identity bounds, metadata inheritance via the
existing annotation tests, setter/remake behavior, mixed grouped transforms,
unbounded explicit transforms, identity differentiation, array sharing, and
transformed forward initialization. The latter exposed and fixed an existing
initialization path that could store a natural guess as an internal value.

Reproduce the workload with each source tree developed in the same benchmark
environment:

```sh
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 julia --project=<bench-env> benchmark/bounded_transforms.jl
OPENBLAS_NUM_THREADS=1 JULIA_NUM_THREADS=1 taskset -c 6 julia --project=<bench-env> benchmark/shared_grids.jl
```
