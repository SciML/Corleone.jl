# OptimalControlBenchmarks

The package provides ModelingToolkit benchmark definitions and a driver using
Corleone's shared optimization bridge. The package README explains normalized
grids, result records, timing scope, and solver settings. `main.jl` is an executable
LQR example; the complete catalog is returned by `load_benchmarks()`.

## Grids and registry

```@docs
BenchmarkGrids
load_benchmarks
OptimalControlBenchmarks.OptimalControlBenchmark
OptimalControlBenchmarks.scale_grids
OptimalControlBenchmarks.scale_grids!
```

## Preparation, solving, and timing

```@docs
prepare_benchmark
solve_with_corleone
run_all
OptimalControlBenchmarks.plot_oc_problem
```

`prepare_benchmark` returns a `CorleoneDynamicOptProblem`. Use its parameters and
state with Corleone's bounds, vectorization, initialization, and optimization APIs.
Controls are parameter histories and the plotting helper selects them through
symbolic metadata. Sampled path constraints do not establish feasibility between
samples. Inspect each optimization result's termination status; successful model
construction does not imply convergence or global optimality.
