# One Julia/BLAS thread; run with this checkout and BenchmarkTools available.
using Corleone, LuxCore, Random, ComponentArrays, OrdinaryDiffEqTsit5, BenchmarkTools, LinearAlgebra

function report(name, f)
    f()
    estimate = median(@benchmark $f() samples = 500 seconds = 1.0)
    println(name, ',', estimate.time, ',', estimate.memory, ',', estimate.allocs)
    return flush(stdout)
end

function compare_experiment(n)
    problem = ODEProblem((u, p, t) -> [-p[1] * u[i] + p[i + 1] for i in eachindex(u)], ones(n), (0.0, 2.0), [0.7; fill(0.2, n)])
    resets = n == 1 ? [0.0] : collect(0.0:0.1:1.9)
    switches = n == 1 ? nothing : collect(0.1:0.1:1.9)
    definitions = (
        Tunable(collect(1:n); kind = Val(:variable), default = ones(n)) => resets,
        Tunable(collect(2:(n + 1)); default = fill(0.2, n)) => switches,
    )
    base = ShootingLayer(problem, Tsit5(), definitions...; controls = collect(2:(n + 1)))
    spec = (annotations = definitions,)
    batch = Experiments(base, (a = spec, b = spec))
    rng = Xoshiro(1)
    ps, st = LuxCore.setup(rng, batch)
    configs = map(keys(batch.experiments)) do name
        config = batch.configurations[name]
        direct = Corleone.experiment_shooting(base, config)
        direct_ps = Corleone.merge_experiment_ps(ps.shared, ps.experiments[name], st.shared, st.experiments[name])
        direct_st = merge(
            Corleone.merge_experiment_st(config, st.shared, st.experiments[name], st.execution[name].active),
            (; timestops = st.execution[name].timestops)
        )
        (; layer = direct, ps = direct_ps, st = direct_st)
    end
    a, b = configs
    reference = first(a.layer(nothing, a.ps, a.st))
    @assert first(base(batch, ps, st))[1].u ≈ reference.u
    @assert first(base(batch, ComponentArray(ps), st))[1].u ≈ reference.u
    base_ps, base_st = LuxCore.setup(Xoshiro(1), base)
    @assert first(base(nothing, base_ps, base_st)).u ≈ reference.u
    report("$(n)_base", () -> base(nothing, base_ps, base_st))
    report("$(n)_direct_one", () -> a.layer(nothing, a.ps, a.st))
    report("$(n)_direct_two", () -> (a.layer(nothing, a.ps, a.st), b.layer(nothing, b.ps, b.st)))
    report("$(n)_batch_two", () -> base(batch, ps, st))
    report("$(n)_merge_native", () -> batch(nothing, ps, st))
    packed = ComponentArray(ps)
    report("$(n)_merge_packed", () -> batch(nothing, packed, st))
    report("$(n)_batch_two_packed", () -> base(batch, packed, st))
    return report("$(n)_setup", () -> LuxCore.setup(rng, batch))
end
println(stderr, "Julia=", VERSION, ", threads=", Threads.nthreads(), ", BLAS=", BLAS.get_num_threads(), ", source=", pkgdir(Corleone))
println("case,median_ns,bytes,allocations")
compare_experiment(1)
compare_experiment(24)
