# OptimalControlBenchmarks

OptimalControlBenchmarks provides 31 ModelingToolkit optimal-control problems and
a driver using Corleone's shared optimization bridge. Use it to prepare, solve,
time, and plot problems with consistent control and shooting grids.

## Solve an LQR problem

The linear quadratic regulator (LQR) benchmark tracks a target state while
penalizing control effort. Start with single shooting and a piecewise-constant
control on 20 intervals:

```@example benchmark_lqr
using OptimalControlBenchmarks, Ipopt

grids = BenchmarkGrids(;
    control_grid = 0.0:0.05:0.95,
    shooting_grid = Float64[],
    constraint_grid = 0.0:0.05:1.0,
)
catalog = load_benchmarks()
lqr = only(filter(problem -> nameof(problem) == :lqr, catalog))
model_kwargs = (; problem_kwargs = (; abstol = 1e-10, reltol = 1e-10))
solution = solve_with_corleone(
    lqr, Ipopt.Optimizer(), grids, false; model_kwargs, print_level = 0
)
(; objective = solution.objective, retcode = solution.retcode)
```

Grids are normalized to `[0, 1]`. Each constructor scales independent copies to
its physical time span, so the same grids can be reused across problems. Control
points mark interval starts; a terminal endpoint is optional and is discarded
when preparing the control schedule. Use `shooting_grid = 0.0:0.1:1.0` for multiple
shooting. Interior shooting points lift states and add continuity constraints;
endpoints do not create additional segments.

!!! note "Initial guesses and tolerances"
    Dense shooting grids may need better starting values. Use `prepare_benchmark`
    to obtain a model, initialize its shooting values with Corleone's
    `forward_initialization(rng, model.layer, model.ps, model.st)`, then construct
    the optimization problem. Set integration tolerances through `problem_kwargs`
    to resolve the requested optimization accuracy.

## Prepare and time problems

`prepare_benchmark` returns a `CorleoneDynamicOptProblem`. Use its parameters and
state with Corleone's bounds, vectorization, initialization, and optimization APIs.
When evaluating optimized parameters, reconstruct them with the chosen vectorizer
and reuse `model.st`. Controls are parameter histories; plotting selects them
through symbolic metadata.

To collect timing records, use the same catalog and grids:

```julia
results = run_all([lqr], Ipopt.Optimizer(), grids; model_kwargs, print_level = 0)
```

Each record contains `name`, `time` in seconds, `objective`, and `retcode`.
Timing includes model construction, first-use compilation during the call, and
optimization. Package import time is excluded and plotting is disabled by default.
These are single-run wall-clock measurements, not warmed statistical benchmarks.
Pass `catalog` instead of `[lqr]` to run all problems.

!!! warning "Interpret results before comparing timings"
    Inspect each solver's termination status. Successful model construction does
    not imply convergence or global optimality. Sampled path constraints do not
    establish feasibility between samples. Nonlinear problems may encounter
    invalid trial states; exceptions propagate from the driver.

`solve_with_corleone` accepts `AD`, `vectorize`, `algorithm`, and `model_kwargs`.
Remaining keywords override the optimizer defaults (`max_iter = 100`,
`tol = 5e-6`, and `hessian_approximation = "limited-memory"`); they must suit the
chosen optimizer. Its fourth positional Boolean controls plotting.

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
