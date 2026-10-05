using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using Optimization, OptimizationMOI, Ipopt, ComponentArrays
using SciMLBase
using Test
using Random
using StableRNGs

rng = StableRNG(505)

# Model: du/dt = -p[1]*u, exact solution u(t) = u0 * exp(-p[1]*t).
f = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
p_true = 0.7
prob = ODEProblem(f, [1.0], (0.0, 5.0), [0.3])
tdata = collect(0.0:0.25:5.0)
ydata = exp.(-p_true .* tdata) .+ 0.01 .* randn(rng, length(tdata))
data = ExperimentData(tdata, ydata; σ = 0.01)
pe = ParameterEstimationLayer(prob, Tsit5(), data; observed = (u, p, t) -> u[1:1])

@testset "Optimization extension: problem construction" begin
    optprob = OptimizationProblem(pe)
    @test optprob isa Optimization.OptimizationProblem
    @test length(optprob.u0) == 1
    @test optprob.lb == problem(pe).lb
    @test optprob.ub == problem(pe).ub

    # The scalar objective is `0.5 * Σ rᵢ²`, evaluated at the initial guess.
    obj0 = optprob.f(optprob.u0, nothing)
    @test isfinite(obj0)
    @test obj0 >= 0

    # A custom initial guess is forwarded.
    optprob2 = OptimizationProblem(pe; u0 = [p_true])
    @test optprob2.u0 == [p_true]
end

@testset "Optimization extension: Ipopt recovers the parameter" begin
    optprob = OptimizationProblem(pe)
    sol = solve(optprob, Ipopt.Optimizer(); print_level = 0, tol = 1.0e-10)
    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.u[1], p_true; atol = 0.05)
end
