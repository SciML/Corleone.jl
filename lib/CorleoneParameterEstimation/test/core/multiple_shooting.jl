using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LuxCore
using Test
using Random
using StableRNGs

rng = StableRNG(99)

# Model: du/dt = -p[1]*u, exact solution u(t) = u0 * exp(-p[1]*t).
f = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
p_true = 0.3
prob = ODEProblem(f, [1.0], (0.0, 5.0), [0.1])   # deliberately wrong parameter guess

tdata = collect(0.0:0.5:5.0)
u_exact = exp.(-p_true .* tdata)
σ = nothing                                  # unweighted residuals (noise estimated from fit)
ydata = u_exact .+ (isnothing(σ) ? 0.0 : σ .* randn(rng, length(tdata)))
data = ExperimentData(tdata, ydata; σ = σ)

# 6 shooting points -> 5 shooting intervals -> 4 local shooting-state variables.
shooting_points = collect(0.0:1.0:5.0)

@testset "multiple shooting layer construction" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data, shooting_points;
        observed = (u, p, t) -> u[1:1])
    @test pe isa ParameterEstimationLayer
    @test pe.layer[1] isa MultipleShootingLayer
    ls, st = LuxCore.setup(Random.default_rng(), pe)
    @test haskey(ls, :interval_1)
    @test length(keys(ls)) == length(shooting_points) - 1

    # Flat optimization variables: the shared model block `[u0; p]` (here just `p`,
    # since there are no tunable initial conditions) plus the local shooting states
    # of intervals 2..K (4 of them: (6 points - 1) * 1 state).
    nlls = problem(pe)
    @test length(nlls.u0) == 1 + 4
    # 11 observations + 4 boundaries * (1 state + 1 parameter) continuity residuals.
    corr = CorleoneParameterEstimation.n_residuals(pe) - CorleoneParameterEstimation.n_observations(data)
    @test corr == 4 * 2
    @test n_shooting_constraints(pe) == corr
end

@testset "GaussNewton fit with multiple shooting" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data, shooting_points;
        observed = (u, p, t) -> u[1:1])
    nlls = problem(pe)
    sol = solve(nlls, GaussNewton(); abstol = 1e-8, reltol = 1e-8)
    @test SciMLBase.successful_retcode(sol)

    res = estimate_parameters(pe, sol)
    # Only the shared model parameter is reported (the local shooting states are
    # profile-eliminated nuisance variables), and it recovers the truth.
    @test res.n_parameters == 1
    @test length(res.x) == 1
    @test isapprox(res.x[1], p_true; atol = 0.05)
end

@testset "LevenbergMarquardt fit with multiple shooting" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data, shooting_points;
        observed = (u, p, t) -> u[1:1])
    nlls = problem(pe)
    sol = solve(nlls, LevenbergMarquardt())
    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.u[1], p_true; atol = 0.1)
end

@testset "fitted trajectory is reconstructed" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data, shooting_points;
        observed = (u, p, t) -> u[1:1])
    sol = solve(problem(pe), GaussNewton(); abstol = 1e-8, reltol = 1e-8)
    ft = fitted_parameters(pe, sol)
    # single experiment -> a single assembled trajectory covering the whole time span
    @test ft.trajectory isa Trajectory
    @test minimum(ft.trajectory.t) <= 0.0
    @test maximum(ft.trajectory.t) >= 5.0
    # the reconstructed end state matches the true solution to good accuracy
    @test isapprox(ft.trajectory.u[end][1], exp(-p_true * 5.0); atol = 0.05)
end

@testset "joint multiple shooting over two experiments" begin
    # Two experiments with different initial conditions share the model parameter but
    # have *independent* local shooting states.
    t = collect(0.0:0.5:5.0)
    y1 = exp.(-p_true .* t) .+ 0.0 .* randn(rng, length(t))
    y2 = 2.0 .* exp.(-p_true .* t) .+ 0.0 .* randn(rng, length(t))
    data2 = ExperimentData(
        ExperimentData(t, y1; σ = σ),
        ExperimentData(t, y2; σ = σ, u0 = [2.0]),
    )
    pe = ParameterEstimationLayer(prob, Tsit5(), data2, shooting_points;
        observed = (u, p, t) -> u[1:1])
    @test n_experiments(pe.data) == 2

    nlls = problem(pe)
    # 1 shared parameter + 2 experiments * 4 local states = 9 optimization variables
    @test length(nlls.u0) == 1 + 2 * 4

    sol = solve(nlls, GaussNewton(); abstol = 1e-8, reltol = 1e-8)
    @test SciMLBase.successful_retcode(sol)
    res = estimate_parameters(pe, sol)
    @test isapprox(res.x[1], p_true; atol = 0.05)

    ft = fitted_parameters(pe, sol)
    @test ft.trajectory isa Tuple             # one reconstructed trajectory per experiment
    @test length(ft.trajectory) == 2
    @test occursin("fitted trajectory: 2 experiments", sprint(show, ft))
end
