# Sparsity preparation review — 2 November 2026

Compared a source snapshot of the sparse-shooting implementation at the start of
this review with the optimized working tree. Julia 1.12.7, Linux, one Julia thread
and one BLAS thread; both checkouts used the same dependency manifest. These are
warmed pattern-preparation measurements, excluding imports, model construction,
first-use compilation, coloring, and numerical differentiation. MB denotes
1,000,000 allocated bytes. Medians use 20 timed calls after warmup; the dense-history
case uses 30. Small samples describe these workloads, not universal speedups.

## Measurements

The shared-grid cases use `u′ = p`, with one tunable state and parameter buffer
per variable. All variables share the switching/reset grid. The parameter-read
cases additionally parse literal vector constraints: either 19 time reads of one
parameter or one time read of each of 128 parameters.

| Pattern preparation | Before | After | Before MB | After MB |
|---|---:|---:|---:|---:|
| 24 states, 19 resets | 1.780 ms | 0.775 ms | 4.74 | 2.29 |
| 96 states, 39 resets | 43.407 ms | 27.855 ms | 144.98 | 69.31 |
| 128 states, 19 resets, selected parameter | 40.512 ms | 23.332 ms | 137.11 | 58.25 |
| 128 states, 19 resets, all parameter reads | 35.686 ms | 24.931 ms | 128.84 | 62.37 |
| Three states, 10,001 saving times, mixed series/timed reads | 21.613 ms | 0.0359 ms | 36.42 | 0.0404 |

The last case has two full resets, three control switches, and constraints
`[sum(u₁), sum(p₁), u₂(1.0), p₂(0.5)]`. The before/after sparse entry coordinates
matched exactly. The large saving grid previously caused a dependency snapshot
and parameter lookups for essentially every saved sample. The optimized walk
records requested state times and whole-history unions, and bounds the sampled
parameter intervals by the first and last saved times.

The first four workloads were measured after allocation cleanup; the final
history fast path was measured separately. Numerical Jacobian callback allocations
remained 3,361,152 bytes in the one-thread Lotka case. No consistent execution-time
benefit was established for an in-place DI variant, whose allocation reduction
was about 0.5%; the existing numerical derivative path was retained.

## Implementation changes

- Reuse the packed vector and reconstructor when preparing sparse callbacks.
- Materialize only the integer column-ID tree once for ComponentArrays; numerical
  parameters keep their existing view/reconstruction behavior.
- Share read-only dependency sets and sample snapshots. Resets replace references;
  they never mutate published sets. Whole-state series share their common union
  under the existing conservative all-state-coupling assumption.
- Reuse a reset-row buffer rather than allocating dictionaries and sorting keys
  at every segment. Read grouped reset columns directly without temporary tuples.
- Resolve only referenced parameter signals; canonical locations follow the
  supplied state permutation. Cache series dependencies separately from timed reads.
- Reuse literal-vector component dependencies and reserve sparse triplet capacity.
- Skip empty Jacobian preparation and avoid per-sample work for bare histories.

`benchmark/sparsity_preparation.jl` provides these preparation workloads. Run it
separately from tests; see `benchmark/README.md`. It retains scalar timing records
rather than keeping every generated matrix alive.

## Correctness review and validation

Regression coverage includes full/partial/fixed resets, free global initial
assignments, unused terminal assignments, noncontiguous shared groups, supplied
parameter permutations, callback initialization, internal saving, branches,
vector/splat expressions, changed fixed values in Lux state, and custom packing.
Sparse entries and Jacobians are checked against dense AD. Additional checks
exclude parameter intervals outside the saved time span.

The broader regression suite passed 395 checks. Final targeted checks covered the
history fast path and two additional span-boundary assertions, alongside sparse
optimization and MTK. Strict root precompilation and loading both numerical
extensions passed. This is not a full platform/version CI result or a fresh
whole-site documentation build.

## Remaining execution-time opportunities

Preparation is cached, so these savings primarily affect model/optimizer setup.
For further numerical gains:

1. Measure hoisting invariant schedule/count validation out of the colored AD
   seed evaluations, while preserving actual saved-layout checks.
2. Use available RHS state/parameter dependencies to refine the current dense
   within-segment coupling assumption. Flow dependencies need transitive closure;
   a sparse RHS Jacobian alone is not the segment solution's sparsity pattern.
