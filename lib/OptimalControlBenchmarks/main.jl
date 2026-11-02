using OptimalControlBenchmarks, Ipopt

# One normalized configuration can be reused across different problem horizons.
grids = BenchmarkGrids(;
    control_grid = 0.0:0.05:0.95,
    shooting_grid = Float64[], # Start with single shooting.
    constraint_grid = 0.0:0.05:1.0,
)

# Start with the LQR benchmark. Use load_benchmarks() for the complete catalog.
benchmarks = filter(benchmark -> nameof(benchmark) == :lqr, load_benchmarks())
# Integration accuracy must resolve the requested optimization tolerance.
model_kwargs = (; problem_kwargs = (; abstol = 1.0e-10, reltol = 1.0e-10))
results = run_all(benchmarks, Ipopt.Optimizer(), grids; model_kwargs, print_level = 0)
println(results)
