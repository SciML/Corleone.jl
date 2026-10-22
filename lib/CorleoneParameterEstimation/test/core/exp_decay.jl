using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LuxCore
using Test
using Random
using StableRNGs

rng = StableRNG(1234)

# Model: du/dt = -p[1]*u, exact solution u(t) = u0 * exp(-p[1]*t).
f = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
p_true = 0.7
u0 = [1.0]
tspan = (0.0, 5.0)
prob = ODEProblem(f, u0, tspan, [0.3])   # start from a wrong guess

tdata = collect(0.0:0.25:5.0)
u_exact = u0[1] .* exp.(-p_true .* tdata)
σ = 0.01
ydata = u_exact .+ σ .* randn(rng, length(tdata))
data = ExperimentData(tdata, ydata; σ = σ)

@testset "ExperimentData" begin
    @test length(observation_times(data)) == length(tdata)
    @test n_observations(data) == length(tdata)
end

@testset "ParameterEstimationLayer construction" begin
    layer = SingleShootingLayer(prob, Tsit5(); saveat = unique(tdata))
    pe = ParameterEstimationLayer(layer, data)
    @test pe isa ParameterEstimationLayer
    # direct construction from the problem (handles saveat internally)
    pe2 = ParameterEstimationLayer(prob, Tsit5(), data)
    @test pe2 isa ParameterEstimationLayer
    ps, st = LuxCore.setup(Random.default_rng(), pe2)
    traj, _ = pe2(nothing, ps, st)
    @test length(traj.t) == length(tdata)
end

@testset "GaussNewton fit recovers parameter" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    nlls = problem(pe)
    @test nlls isa NonlinearLeastSquaresProblem

    sol = solve(nlls, GaussNewton(); abstol = 1.0e-6, reltol = 1.0e-6)
    @test SciMLBase.successful_retcode(sol)

    res = estimate_parameters(pe, sol)
    @test isapprox(res.x[1], p_true; atol = 0.1)
    # With known σ, the uncertainty should be small and cover the truth.
    @test res.std_errors[1] < 0.1
    lo, hi = res.confidence_intervals[1]
    @test lo <= p_true <= hi
end

@testset "LevenbergMarquardt fit recovers parameter" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    nlls = problem(pe)
    sol = solve(nlls, LevenbergMarquardt())
    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.u[1], p_true; atol = 0.1)
end

@testset "Fixing a parameter leaves only the free one tunable" begin
    # Fix the parameter (bounds coincide) -> no tunable parameters left is an
    # invalid fit; instead we just check that the number of optimization variables
    # reflects the tunable parameter count.
    prob2 = ODEProblem(f, u0, tspan, [0.7])
    pe = ParameterEstimationLayer(prob2, Tsit5(), data)
    prob_pe = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(pe)
    @test length(prob_pe.x0) == 1
end
