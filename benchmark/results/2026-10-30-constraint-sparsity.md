# Structural constraint sparsity — 30 October 2026

Root Corleone, Julia 1.12.7 on Linux, four Julia threads and one BLAS thread.
Dependencies: SciMLBase 3.58.0, OrdinaryDiffEqTsit5 2.1.5, ForwardDiff 1.4.6,
DifferentiationInterface 0.7.21, SparseMatrixColorings 0.4.29, ComponentArrays
0.15.50, Optimization 5.9.1, OptimizationBase 5.10.0, OptimizationMOI 1.6.0,
and Ipopt 1.16.0.

The numerical model is the multiple-shooting Lotka fishing tutorial, with Tsit5,
`abstol=reltol=1e-9`, three interior full-state resets, 48 original control
intervals and two inserted intervals. There are 59 tunable values and 11 matching
constraints. Ipopt uses `tol=1e-6`, `max_iter=1000`, and limited-memory Hessians.

`constraint_jacobian_sparsity` generated exactly the same pattern as the earlier
manual prototype: 154 entries in an 11 × 59 matrix, with 19 column colors.
No numerical derivative threshold is used to generate the pattern. Sparse
Jacobians were compared with dense ForwardDiff at four parameter inputs and at
the optimized solutions. Matching residuals were below `1e-6`.

| Measurement | Dense | Sparse constraints |
|---|---:|---:|
| Warm Jacobian median | 1.004155 ms | 0.445125 ms |
| Warm Jacobian allocated bytes | 8,944,088 | 3,360,512 |
| Warm solve median | 1.141681 s | 0.641341 s |
| Warm solve allocated bytes | 5,799,153,160 | 3,081,734,896 |
| Objective | 1.3456328025 | 1.3456313298 |
| Solver status | Success | Success |

Jacobian medians use 30 warmed in-place evaluations. Solve medians use three calls after an initial solve per configuration. These are
small local samples, not a universal speedup claim. Package loading, model
construction, and sparse preparation are excluded from warmed solve times.
Both configurations start from the same parameter values. Small differences in
solver paths produce slightly different final objectives within the requested
optimization accuracy.

The objective retains dense ForwardDiff. Only the constraint Jacobian uses DI's
`AutoSparse`, a known structural pattern, and SparseMatrixColorings. This avoids
both tracing through adaptive integration and requesting Hessian sparsity from a
Jacobian-only detector. The prepared Jacobian is passed through `cons_j` with a
sparse `cons_jac_prototype`.

Reproduce using `benchmark/shooting_sparse.jl` in a separate prepared benchmark
environment; see `benchmark/README.md`. It validates derivatives, reports warmed
Jacobian timings and allocations, and repeats the dense and sparse optimizations.
The benchmark excludes plots and the tutorial's separate single-shooting solve.
