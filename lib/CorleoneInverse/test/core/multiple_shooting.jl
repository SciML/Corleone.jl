using Test, CorleoneInverse, OrdinaryDiffEqTsit5, ComponentArrays, NonlinearSolve, LuxCore, StableRNGs
using Optimization, OptimizationMOI, Ipopt

function lotka(u, p, t)
    return [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[2] * u[1] * u[2]]
end
θ = [1.5, 1.0, 3.0]
problem = ODEProblem(lotka, [1.0, 1.0], (0.0, 5.0), θ; abstol = 1.0e-10, reltol = 1.0e-10)
t = collect(0.25:0.25:5.0)
noise = 0.02 .* sin.(1:(2 * length(t)))
y = reduce(hcat, solve(problem, Tsit5(); saveat = t).u) .+ reshape(noise, 2, :)
experiment = Experiment(:u₁ => (t, y[1, :], 0.02), :u₂ => (t, y[2, :], 0.02))
parameters = (
    Tunable(1; kind = Val(:parameter), default = 1.4, lowerbound = 0.0) => nothing,
    Constant(2; kind = Val(:parameter)) => nothing,
    Tunable(3; kind = Val(:parameter), default = 2.8, lowerbound = 0.0) => nothing,
)
# Initialize the shooting node from the measurement at the node time.
node = y[:, findfirst(==(2.5), t)]
nodes = (
    Tunable(1; kind = Val(:variable), default = node[1], lowerbound = 0.0) => [2.5],
    Tunable(2; kind = Val(:variable), default = node[2], lowerbound = 0.0) => [2.5],
)
single = ShootingLayer(problem, Tsit5(), parameters...)
multiple = ShootingLayer(problem, Tsit5(), parameters..., nodes...)

ps, st = LuxCore.setup(StableRNG(1), single)
reference = solve(NonlinearLeastSquaresProblem(single, experiment, ps, st), LevenbergMarquardt())
reference_estimate = ParameterEstimate(single, experiment, reference.u, ps, st)

@testset "Multiple shooting with matching constraints" begin
    ps, st = LuxCore.setup(StableRNG(1), multiple)
    @test_throws ArgumentError NonlinearLeastSquaresProblem(multiple, experiment, ps, st)
    @test_throws ArgumentError OptimizationProblem(multiple, (experiment, experiment), ps, st, Optimization.AutoForwardDiff())
    optprob = OptimizationProblem(multiple, experiment, ps, st, Optimization.AutoForwardDiff())
    @test optprob.lb == zeros(4)
    @test optprob.lcons == optprob.ucons == zeros(2)
    sol = solve(optprob, Ipopt.Optimizer(); print_level = 0, tol = 1.0e-10, hessian_approximation = "limited-memory")
    est = ParameterEstimate(multiple, experiment, sol.u, ps, st)
    fitted = ComponentArray(sol.u, getaxes(ComponentArray(ps)))
    @test fitted.parameter.parameters.p₁.parameter.p₁.ps ≈ reference.u[1:1] rtol = 1.0e-6
    @test fitted.parameter.parameters.p₁.parameter.p₃.ps ≈ reference.u[2:2] rtol = 1.0e-6
    @test est.dof == reference_estimate.dof
    @test est.rss ≈ reference_estimate.rss rtol = 1.0e-6
    # The covariance restricted to the matching constraints equals the single-shooting covariance.
    @test est.covariance[3:4, 3:4] ≈ reference_estimate.covariance rtol = 1.0e-4
end
