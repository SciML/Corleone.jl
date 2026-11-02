# Run separately from other timings; see benchmark/README.md.
using Corleone, ComponentArrays, LuxCore, OrdinaryDiffEqTsit5, Random
using Optimization, OptimizationMOI, Ipopt, ForwardDiff, ADTypes, SparseArrays, Statistics
import DifferentiationInterface as DI
import SparseMatrixColorings

function lotka_dynamics(du, u, p, t)
    du[1] = u[1] - p[1] * prod(u[1:2]) - 0.4 * p[2] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[2] * u[2]
    return du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
end
tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [1.0, 0.0, 1.0]
prob = ODEProblem(lotka_dynamics, u0, tspan, p0; abstol = 1.0e-9, reltol = 1.0e-9)
cgrid = collect(0.1:0.1:11.9)
control = Tunable(
    2; kind = Val(:parameter), default = 1.0,
    lowerbound = 0.0, upperbound = 1.0
) => cgrid
physical_parameters = (
    Constant([3, 1]; kind = Val(:parameter)) => nothing,
)

shooting_points = [3.4, 6.1, 9.75]
initial_conditions = (
    Tunable(i; kind = Val(:variable), lowerbound = 0.0, upperbound = 2.0) => shooting_points for i in 1:3
)
mslayer = ShootingLayer(
    prob, Tsit5(), first(control) => collect(0.25:0.25:11.75), physical_parameters...,
    initial_conditions...; ensemblealg = EnsembleThreads(),
    controls = [2], quadratures = [3]
)
msps, msst = LuxCore.setup(MersenneTwister(42), mslayer)
msps, msst = add_parameter_segments!(mslayer, msps, msst)
ms_dynprob = CorleoneDynamicOptProblem(
    mslayer, :(u₃(12.0)), ps = msps, st = msst
)

# Compare the existing limited-memory setup with dense and colored exact Hessians.
q, reconstruct = pack(default_vectorizer(), msps)
pattern = lagrangian_hessian_sparsity(ms_dynprob)
println("Julia=", VERSION, ", threads=", Threads.nthreads(), ", variables=", length(q), ", constraints=", length(ms_dynprob.lcons))
println("HESSIAN_PATTERN,entries=", nnz(pattern), ",dense_entries=", length(q)^2)
flush(stdout)
lagrangian = (x, sigma, lambda, st) -> Corleone.optimization_lagrangian(ms_dynprob, reconstruct(x), st, sigma, lambda)
lambda = ones(eltype(q), length(ms_dynprob.lcons))
contexts = (DI.Constant(1.0), DI.Constant(lambda), DI.Constant(msst))
sparse_backend = AutoSparse(AutoForwardDiff(); sparsity_detector = ADTypes.KnownHessianSparsityDetector(pattern), coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm())
for (name, backend) in (("dense", AutoForwardDiff()), ("sparse", sparse_backend))
    measured = @timed DI.prepare_hessian(lagrangian, backend, q, contexts...)
    prep = measured.value
    println("PREPARE,", name, ",seconds=", measured.time, ",bytes=", measured.bytes)
    if name == "sparse"
        println("HESSIAN_COLORS,", maximum(SparseMatrixColorings.column_colors(prep); init = 0))
    end
    H = DI.hessian(lagrangian, prep, backend, q, contexts...)
    reference = ForwardDiff.hessian(x -> lagrangian(x, 1.0, lambda, msst), q)
    @assert isapprox(Matrix(H), reference; rtol = 1.0e-7, atol = 1.0e-7)
    samples = [@timed DI.hessian!(lagrangian, H, prep, backend, q, contexts...) for _ in 1:5]
    println("HESSIAN,", name, ",median_s=", median(x.time for x in samples), ",bytes=", median(x.bytes for x in samples))
    flush(stdout)
end
for mode in ("limited-memory", "dense-exact", "sparse-exact")
    build = @timed OptimizationProblem(ms_dynprob, AutoForwardDiff(); sparse_constraints = true, sparse_hessian = mode == "sparse-exact")
    opt = build.value
    println("BUILD,", mode, ",seconds=", build.time, ",bytes=", build.bytes)
    times = Float64[]
    for run in 1:4
        options = mode == "limited-memory" ? (; hessian_approximation = "limited-memory") : (;)
        measurement = @timed solve(opt, Ipopt.Optimizer(); tol = 1.0e-6, max_iter = 1000, print_level = 0, options...)
        sol = measurement.value
        residual = maximum(abs, optimization_constraints(ms_dynprob, reconstruct(sol.u)); init = 0.0)
        @assert successful_retcode(sol)
        @assert residual < 1.0e-6
        println("SOLVE,", mode, ",run=", run, ",seconds=", measurement.time, ",bytes=", measurement.bytes, ",objective=", sol.objective, ",residual=", residual, ",stats=", sol.stats)
        flush(stdout)
        run > 1 && push!(times, measurement.time)
    end
    println("WARM_MEDIAN,", mode, ",", median(times))
    flush(stdout)
end
