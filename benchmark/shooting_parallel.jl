# Run in an environment containing Corleone, LuxCore, OrdinaryDiffEqTsit5 and
# BenchmarkTools, e.g. julia --project=<bench-env> --threads=4 benchmark/shooting_parallel.jl
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, BenchmarkTools

function parallel_benchmark_case(heavy)
    function rhs!(du, u, p, t)
        forcing = 0.0
        if heavy
            for k in 1:256
                forcing += sin(t + k * u[1]) / k
            end
        end
        du[1] = p[1] * u[1] + 0.001 * forcing
        return nothing
    end
    prob = ODEProblem(rhs!, [0.5], (0.0, 12.0), [0.1]; abstol = 1.0e-9, reltol = 1.0e-9)
    serial = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => collect(1.0:11.0),
        Tunable(1; kind = Val(:parameter)) => collect(0.1:0.1:11.9)
    )
    parallel = ShootingLayer(
        serial.name, prob, serial.alg, EnsembleThreads(),
        serial.initialcondition, serial.parameter
    )
    ps, st = LuxCore.setup(MersenneTwister(1), serial)
    return serial, parallel, ps, st
end

println("Julia threads: ", Threads.nthreads())
println("case,mode,median_us,bytes,allocations")
for heavy in (false, true)
    serial, parallel, ps, st = parallel_benchmark_case(heavy)
    a, _ = serial(nothing, ps, st)
    b, _ = parallel(nothing, ps, st)
    @assert a.u ≈ b.u
    @assert a.residuals ≈ b.residuals
    for (mode, layer) in (("serial", serial), ("threads", parallel))
        trial = @benchmark $layer(nothing, $ps, $st) samples = 500 evals = 1 seconds = 10
        stats = median(trial)
        println(
            heavy ? "expensive_rhs" : "cheap_rhs", ",", mode, ",",
            stats.time / 1000, ",", stats.memory, ",", stats.allocs
        )
    end
end
