# Shooting benchmarks

Use a separate Julia environment with this checkout developed as `Corleone`.
Benchmarks do not install dependencies or change the package environment.

- `benchmarks.jl`: BenchmarkTools suite for construction, setup, control lookup,
  and single/multiple-shooting forward calls. It retains the legacy constructor
  branch so AirspeedVelocity can run the same workload against the old API.
  Requires BenchmarkTools, OrdinaryDiffEqTsit5, StableRNGs, and LuxCore.
- `forward_comparison.jl`: first-call and warmed forward measurements using the
  same single/multiple-shooting models as `benchmarks.jl`, including endpoint output
  for comparison across checkouts. Requires the same dependencies as that suite.
- `initialization_latency.jl`: a fresh-process first-use measurement of annotation,
  setup, reset, and lookup operations. Run after precompiling each checkout;
  package loading is excluded. A second call reports the warmed cost. Requires
  SciMLBase, LuxCore, and Random alongside Corleone.
- `shooting_parallel.jl`: warmed serial/threaded comparisons for cheap and expensive
  ODE right-hand sides. Requires BenchmarkTools, LuxCore, OrdinaryDiffEqTsit5, and
  Random alongside Corleone.
- `shooting_ad.jl`: forward loss and prepared gradient timings for a Lotka ODE,
  using controls alone or joint state/parameter shooting variables. Requires
  BenchmarkTools, LuxCore, OrdinaryDiffEqTsit5, ComponentArrays,
  DifferentiationInterface, ADTypes, Random, and the selected backend.

Run timing workloads separately from tests and other benchmarks. Set
`OPENBLAS_NUM_THREADS=1`; use one Julia thread for backend comparisons and several
for the parallel workload. For example:

```sh
OPENBLAS_NUM_THREADS=1 julia --project=<bench-env> --threads=4 benchmark/shooting_parallel.jl
OPENBLAS_NUM_THREADS=1 julia --project=<bench-env> --threads=1 benchmark/shooting_ad.jl forwarddiff
```

Run each AD backend in a fresh process. Supported script arguments are
`forwarddiff`, `reversediff`, `reversediff_default`, `zygote`, `mooncake`,
`enzyme`, and `finitediff`. Zygote, ReverseDiff, Mooncake, and Enzyme also require
SciMLSensitivity. The default ReverseDiff route and Enzyme are diagnostic cases;
being selectable does not establish compatibility. Errors are reported as `FAIL`
rows with full traces in the system temporary directory.

The AD workload uses Tsit5, three states, a time span of `[0, 4]`, eight control
intervals, and absolute/relative tolerances of `1e-10`. The joint case adds full
state resets at `1.25` and `2.75` and splits crossing parameter intervals. The loss
is the terminal quadrature plus `0.1 * sum(abs2, shooting_constraints(trajectory))`.
Each successful gradient is checked at two inputs against central differences,
including reuse of its preparation and checks that inputs remain unchanged.

`FORWARD` rows report first loss-call seconds, warmed median milliseconds, bytes,
and allocations. `RESULT` rows report first gradient preparation plus evaluation
seconds, warmed median prepared `gradient!` milliseconds, bytes, allocations, and
maximum gradient error. Warmed trials use up to 100 samples, one evaluation per
sample, and a ten-second sampling budget. Package loading and layer construction
are excluded. Gradient preparation follows the forward measurement. The joint
case runs after the controls case, so its first timings reuse compilation from
that case; they are incremental first-call costs, not isolated cold starts.
Forward loss timings include parameter reconstruction and trajectory/constraint
construction. They are not just the internal solver time.

Dated measurements and validation results are kept in [`results/`](results/).

First forward and gradient calls use `Base.invokelatest` as an inference barrier
so specialization occurs inside the timer. Warmed trials call the functions directly.
