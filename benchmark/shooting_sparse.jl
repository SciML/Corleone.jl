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

q, reconstruct = pack(default_vectorizer(), msps)
pattern = constraint_jacobian_sparsity(ms_dynprob)
f = q -> optimization_constraints(ms_dynprob, reconstruct(q), msst)
dense = AutoForwardDiff()
sparse_ad = AutoSparse(
    dense; sparsity_detector = ADTypes.KnownJacobianSparsityDetector(pattern),
    coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
)
println("Julia=", VERSION, ", threads=", Threads.nthreads())
println("pattern rows=", size(pattern, 1), " columns=", size(pattern, 2), " entries=", nnz(pattern))
for (name, backend) in (("dense", dense), ("sparse", sparse_ad))
    prep = DI.prepare_jacobian(f, backend, q)
    if name == "sparse"
        println("colors=", maximum(SparseMatrixColorings.column_colors(prep)))
    end
    for shift in (0.0, 0.001, 0.01, 0.05)
        input = q .+ shift .* sin.(eachindex(q))
        @assert isapprox(Matrix(DI.jacobian(f, prep, backend, input)), DI.jacobian(f, dense, input); rtol = 1.0e-8, atol = 1.0e-9)
    end
    output = DI.jacobian(f, prep, backend, q)
    DI.jacobian!(f, output, prep, backend, q)
    samples = [@timed DI.jacobian!(f, output, prep, backend, q) for _ in 1:30]
    println("JACOBIAN,", name, ",median_s=", median(x.time for x in samples), ",bytes=", median(x.bytes for x in samples))
    opt = OptimizationProblem(ms_dynprob, dense; sparse_constraints = name == "sparse")
    warm_times = Float64[]
    for run in 1:4
        measurement = @timed solve(opt, Ipopt.Optimizer(); tol = 1.0e-6, hessian_approximation = "limited-memory", max_iter = 1000, print_level = 0)
        sol = measurement.value
        @assert successful_retcode(sol)
        @assert maximum(abs, f(sol.u)) < 1.0e-6
        @assert isapprox(Matrix(DI.jacobian(f, prep, backend, sol.u)), DI.jacobian(f, dense, sol.u); rtol = 1.0e-7, atol = 1.0e-8)
        println("SOLVE,", name, ",run=", run, ",seconds=", measurement.time, ",bytes=", measurement.bytes, ",objective=", sol.objective)
        flush(stdout)
        run > 1 && push!(warm_times, measurement.time)
    end
    println("WARM_MEDIAN,", name, ",", median(warm_times))
end
