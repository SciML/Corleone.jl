# Uses the same model and API translation branch for this checkout and legacy main.
include("benchmarks.jl")

function measure_forward(name, layer, ps, st)
    first_call = @timed Base.invokelatest(layer, nothing, ps, st)
    trajectory = first(first_call.value)
    trial = @benchmark $layer(nothing, $ps, $st) samples = 500 evals = 1 seconds = 10
    stats = median(trial)
    println(name, ",", first_call.time, ",", stats.time / 1.0e6, ",", stats.memory, ",", stats.allocs)
    println("endpoint,", name, ",", join(last(trajectory.u), ","))
    return flush(stdout)
end

println("Julia=", VERSION, ", threads=", Threads.nthreads())
println("case,first_seconds,median_ms,bytes,allocations")
measure_forward("single", layer, ps, st)
measure_forward("multiple", mlayer, mps, mst)
