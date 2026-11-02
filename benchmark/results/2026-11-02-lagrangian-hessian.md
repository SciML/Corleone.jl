# Lotka multiple-shooting Hessians

Measured on Julia 1.12.7 with four Julia threads and one OpenBLAS thread.
The problem has 59 optimization variables and 11 matching constraints, with
control intervals of length 0.25 and shooting resets at 3.4, 6.1, and 9.75.
All optimization modes use the same sparse constraint Jacobian and Ipopt
tolerance of `1e-6`.

## Warm optimization results

Times are medians of three solves after an initial solve. Allocations are
cumulative allocated bytes per warmed solve, expressed in decimal GB; they
are not peak memory usage.

| Hessian mode | Solve time | Allocated | Iterations |
|:--|--:|--:|--:|
| Limited-memory | 0.657 s | 3.082 GB | 229 |
| Dense exact | 6.605 s | 38.110 GB | 48 |
| Sparse exact | 1.249 s | 6.733 GB | 48 |

Sparse exact Hessians make the exact-Hessian solve **5.29 times faster** and
reduce allocations by about 82%. Limited-memory remains about **1.90 times
faster** than sparse exact on this example. Exact curvature reduces iteration
count, but its derivative cost outweighs that reduction here.

All solves succeeded. Final objectives were 1.34563133 for limited-memory and
1.34563316 for both exact modes. Maximum absolute constraint residuals were
`7.29e-11`, `1.70e-7`, and `1.70e-7`, respectively.

The sparse implementation evaluates objective and constraints from one shared
trajectory. The standard dense exact path uses OptimizationBase's generated
callbacks, so the full solve comparison includes this implementation difference
in addition to coloring.

## Hessian evaluation alone

Both microbenchmarks differentiate the same Lagrangian with one shared
trajectory, objective weight one, and constraint multipliers all one. Reported
times are medians of five warmed evaluations; results were checked against a
dense ForwardDiff reference.

| Backend | Hessian time | Allocated per evaluation |
|:--|--:|--:|
| Dense ForwardDiff | 107.476 ms | 390.245 MB |
| Colored ForwardDiff | 22.497 ms | 125.803 MB |

Coloring gives a **4.78 times** speedup in this comparison. The conservative
Hessian pattern has 889 stored entries out of 3,481 matrix entries and uses
18 colors.

Preparation took 0.106 s for dense differentiation and 0.733 s for sparse
differentiation. Initial solves took 23.291 s, 23.279 s, and 20.573 s for
limited-memory, dense exact, and sparse exact. These timings are incremental
measurements in one process, in that order, with shared compilation caches;
they are **not independent cold-start or compilation comparisons**.

## Reproduction and validation

Run in an environment containing the checkout and the dependencies listed in
`benchmark/README.md`:

```sh
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 \
julia --startup-file=no --compiled-modules=existing \
--project=/tmp/corleone-sparsity-review/after benchmark/shooting_sparse_hessian.jl
```

Dependency versions: Optimization 5.9.1, OptimizationBase 5.10.0,
OptimizationMOI 1.6.0, DifferentiationInterface 0.7.21,
SparseMatrixColorings 0.4.29, ForwardDiff 1.4.6, Ipopt 1.16.0, and
SciMLBase 3.58.0. The temporary project above is the local validation environment;
substitute your environment path when reproducing.

The following targeted validation passed all **114 assertions**, including
objective and Lagrangian Hessian values, canonical coordinate ordering,
updated state and multiplier values, unconstrained objectives, Ipopt solves,
and existing sparse-Jacobian tests:

```sh
JULIA_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 \
julia --startup-file=no --compiled-modules=existing \
--project=/tmp/corleone-sparsity-review/after \
-e 'include("test/examples/lagrangian_structure.jl"); include("test/examples/sparse_constraints.jl"); include("test/core/lagrangian_sparsity.jl")'
```

The benchmark ran separately after these tests completed. The complete package
suite and documentation build were not run for this numerical integration.
