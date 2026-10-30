benchmark_name(benchmark) = nameof(typeof(benchmark))
benchmark_name(benchmark::Function) = nameof(benchmark)
benchmark_name(benchmark::OptimalControlBenchmark) = benchmark.name

"""
$(SIGNATURES)

Run each benchmark once and return records with `name`, elapsed `time` in seconds,
`objective`, and `retcode`. Timing includes model construction, compilation, and
optimization; it is not a warmed microbenchmark. Inspect solver statuses before
comparing objective values. Exceptions propagate to the caller.

Plotting is disabled by default. Other keywords are forwarded to
[`solve_with_corleone`](@ref). The normalized input grids are reusable and remain
unchanged throughout the run.
"""
function run_all(benchmarks, optimizer, grids::BenchmarkGrids; plot_res::Bool = false, kwargs...)
    return map(collect(benchmarks)) do benchmark
        measured = @timed solve_with_corleone(benchmark, optimizer, grids, plot_res; kwargs...)
        solution = measured.value
        (;
            name = benchmark_name(benchmark), time = measured.time,
            objective = solution.objective, retcode = solution.retcode,
        )
    end
end
