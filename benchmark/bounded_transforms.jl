# Compare this same workload against the pre-refactor and current checkouts in
# separate processes, using one dependency environment and no concurrent tests.
# Package loading and compilation are excluded from the warmed measurements.
using BenchmarkTools, Corleone, LuxCore, Random, OrdinaryDiffEqTsit5, LinearAlgebra
using TransformVariables: asℝ₊
using Distributions: Normal

function report(label, f)
    f()
    trial = @benchmark $f() samples=500 seconds=0.6
    result = median(trial)
    println(label, ',', result.time, ',', result.memory, ',', result.allocs)
end
function bench()
    rng = Xoshiro(1)
    plain = Corleone.PiecewiseConstant(nothing, 2.0; bounds = (0.0, 4.0))
    transformed = Corleone.PiecewiseConstant(nothing, 2.0; transform = asℝ₊)
    a, ast = LuxCore.setup(rng, plain)
    b, bst = LuxCore.setup(rng, transformed)
    @assert first(plain(0.2, a, ast)) == 2.0
    @assert first(transformed(0.2, b, bst)) ≈ 2.0
    report("identity_lookup", () -> plain(0.2, a, ast))
    report("transformed_lookup", () -> transformed(0.2, b, bst))
    values = ones(100)
    pc = Corleone.ParameterCollection((rng, n) -> values, length(values))
    @assert LuxCore.initialparameters(rng, pc).ps === values
    report("identity_array_setup", () -> LuxCore.initialparameters(rng, pc))
    prob = ODEProblem((u, p, t) -> [-p[1] * u[1]], [1.0], (0.0, 2.0), [2.0])
    report("annotation_construct", () -> Tunable(1; lowerbound = 0.0, upperbound = 4.0))
    selectors, lo, hi, maps = [1, 2], [0.0, 0.0], [4.0, 4.0], [nothing, asℝ₊]
    report("grouped_annotation_construct", () -> Tunable(selectors; lowerbound = lo, upperbound = hi))
    report("mixed_annotation_construct", () -> Tunable(selectors; lowerbound = lo, upperbound = hi, transform = maps))
    for (label, transform) in (("identity", nothing), ("transformed", asℝ₊))
        layer = ShootingLayer(prob, Tsit5(), Tunable(1; kind = Val(:parameter), lowerbound = 0.0, upperbound = 4.0, transform, prior = Normal(2, 1)) => nothing)
        println(stderr, "STORAGE,", label, ",", Base.summarysize(layer))
        report(label * "_construct", () -> ShootingLayer(prob, Tsit5(), Tunable(1; kind = Val(:parameter), lowerbound = 0.0, upperbound = 4.0, transform, prior = Normal(2, 1)) => nothing))
        ps, st = LuxCore.setup(rng, layer)
        @assert isapprox(last(first(layer(nothing, ps, st)).u)[1], exp(-4.0); rtol = 5.0e-3)
        report(label * "_setup", () -> LuxCore.setup(rng, layer))
        report(label * "_bounds", () -> get_bounds(layer, ps, st))
        report(label * "_prior", () -> log_prior(layer, ps))
        report(label * "_shooting", () -> layer(nothing, ps, st))
    end
end
println(stderr, "ENV,julia=", VERSION, ",threads=", Threads.nthreads(), ",blas_threads=", BLAS.get_num_threads(), ",cpu=", first(Sys.cpu_info()).model)
println("case,median_ns,bytes,allocations")
bench()
