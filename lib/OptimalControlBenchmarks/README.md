# OptimalControlBenchmarks.jl

A catalog of 31 ModelingToolkit optimal-control problems and a Corleone driver
for preparing, solving, timing, and plotting them. The model equations and costs
are defined in `src/problems/`; the driver uses Corleone's current MTK interface,
parameter vectorizer, bounds, and shooting constraints.

```julia
using OptimalControlBenchmarks, Ipopt

grids = BenchmarkGrids(;
    control_grid = 0.0:0.05:0.95,
    shooting_grid = Float64[],       # Single shooting.
    constraint_grid = 0.0:0.05:1.0,
)
catalog = load_benchmarks()
lqr = only(filter(problem -> nameof(problem) == :lqr, catalog))
model_kwargs = (; problem_kwargs = (; abstol=1e-10, reltol=1e-10))
solution = solve_with_corleone(lqr, Ipopt.Optimizer(), grids, false; model_kwargs, print_level=0)
```

Grids are normalized to `[0, 1]`. Every constructor returns independently scaled
copies, so the same configuration can be reused across different horizons.
Control grids specify interval starts; a terminal endpoint is optional and is
discarded when preparing the control schedule. Interior shooting points lift states
and add matching constraints; endpoints do not create additional shooting segments.
The constraint grid selects sampled path constraints and does not guarantee
continuous-time feasibility between samples.

Use `shooting_grid=0.0:0.1:1.0` for multiple shooting. The MTK interface preserves
fixed initial states while making interior shooting values tunable. Dense shooting
grids can need better starting values: prepare the model, then call Corleone's
`forward_initialization(rng, model.layer, model.ps, model.st)` before constructing
the optimization problem. This updates numerical guesses without rebuilding the
prepared constraint layout.

## Prepare and inspect a model

```julia
using Corleone, ComponentArrays, Optimization, OrdinaryDiffEqTsit5
model = prepare_benchmark(lqr, grids; algorithm=Tsit5(),
    problem_kwargs=(; abstol=1e-10, reltol=1e-10))
problem = OptimizationProblem(model, AutoForwardDiff())
solution = solve(problem, Ipopt.Optimizer(); hessian_approximation="limited-memory")
_, reconstruct = pack(default_vectorizer(), model.ps)
trajectory, _ = model.layer((model.layer.problem, model.evaluation_grid),
    reconstruct(solution.u), model.st)
trajectoryplot(trajectory)
```

Controls remain parameter histories. Plotting uses symbolic metadata rather than
assuming controls are appended to the state vector. Reuse `model.st` when evaluating
optimized parameters; do not initialize a new state or rely on optimizer internals.

## Run and time a catalog

```julia
results = run_all([lqr], Ipopt.Optimizer(), grids; model_kwargs, print_level=0)
# Use `catalog` to run all problems. Plotting is disabled by default.
```

Each record contains `name`, `time` (wall-clock seconds), `objective`, and `retcode`.
Timing includes model construction, first-use compilation during the call, and
optimization; package import time is excluded. These are single-run measurements,
not warmed statistical benchmarks. Inspect solver statuses before comparing
objectives. Nonlinear problems can fail to converge or encounter invalid trial
states; exceptions propagate rather than being recorded as successful runs.

`solve_with_corleone` accepts `AD`, `vectorize`, `algorithm`, and `model_kwargs`.
Remaining keywords override solver defaults (`max_iter=100`, `tol=5e-6`, and
`hessian_approximation="limited-memory"`). The keyword set must suit the chosen
optimizer. Set integration tolerances through `problem_kwargs` to resolve the
requested optimization accuracy; the driver otherwise inherits the ODE solver
defaults. Plotting is controlled by its fourth positional Boolean argument.

## Catalog and development

`load_benchmarks()` returns constructors in alphabetical filename order. Problem
files are discovered and included once at package load time. Repeated registry
queries return independent lists without redefining methods.

A constructor accepts `BenchmarkGrids` and returns `(; system, grids, dims, tspan,
name)`. `dims` counts declared state equations and control signals; automatically
introduced quadrature states are excluded. The driver uses `tspan` explicitly.
Older custom constructors without this field may use the MTK interface's horizon
inference. Keep benchmark equations and scientific assumptions separate from solver
settings and report any mathematical changes explicitly.

The rocket duration starts at `T=0.01` to avoid fuel depletion in the nominal
trajectory; its tunability and feasible bounds are unchanged.

Core tests cover all symbolic constructors, grid isolation, the current solver
bridge, gradients, plotting, and an LQR comparison against an exact
piecewise-constant-control reference. They do not assert that every catalog problem
reaches its global optimum with the default solver settings.

The standalone API documentation is built from `docs/`. Develop the checked-out
Corleone and OptimalControlBenchmarks packages in that environment before running
`julia --project=docs docs/make.jl` from this package directory.
