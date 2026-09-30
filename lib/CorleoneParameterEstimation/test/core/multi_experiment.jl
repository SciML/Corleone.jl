using Corleone, CorleoneParameterEstimation
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LuxCore
using Test
using Random
using StableRNGs

rng = StableRNG(42)

# Model: du/dt = -p[1]*u, exact solution u(t) = u0 * exp(-p[1]*t).
f = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
p_true = 0.7
prob = ODEProblem(f, [1.0], (0.0, 5.0), [0.3])
σ = 0.05

# Two experiments with different initial conditions and different time grids, both
# generated with the same model parameters.
t1 = collect(0.0:0.25:2.5)
y1 = exp.(-p_true .* t1) .+ σ .* randn(rng, length(t1))
t2 = collect(0.0:0.5:5.0)
y2 = 2.0 .* exp.(-p_true .* t2) .+ σ .* randn(rng, length(t2))

data = ExperimentData(
    ExperimentData(t1, y1; σ = σ),               # uses the problem's initial conditions
    ExperimentData(t2, y2; σ = σ, u0 = [2.0]),   # own initial conditions
)

@testset "ExperimentData with multiple experiments" begin
    @test n_experiments(data) == 2
    @test n_observations(data) == length(t1) + length(t2)
    @test observation_times(data, 1) == t1
    @test observation_times(data, 2) == t2
    @test isnothing(initial_conditions(data, 1))
    @test initial_conditions(data, 2) == [2.0]
    # a single experiment keeps its previous behaviour
    data_single = ExperimentData(t1, y1; σ = σ)
    @test n_experiments(data_single) == 1
    @test n_observations(data_single) == length(t1)
    @test observation_times(data_single) == t1
end

@testset "Joint fit over two experiments" begin
    pe = ParameterEstimationLayer(prob, Tsit5(), data)
    @test n_experiments(pe.data) == 2

    # one trajectory per experiment, sampled at the experiment's observation times and
    # started at the experiment's initial conditions
    ps, st = LuxCore.setup(rng, pe)
    out, _ = pe(nothing, ps, st)
    traj1, traj2 = out
    # each trajectory is sampled at that experiment's observation times (it may contain the
    # end of the shared tspan as an extra point, which the residuals crop away)
    @test all(in(traj1.t), t1)
    @test all(in(traj2.t), t2)
    @test first(traj1.u[1]) ≈ 1.0
    @test first(traj2.u[1]) ≈ 2.0

    nlls = problem(pe)
    @test length(nlls.u0) == 1   # only the model parameter is tunable
    sol = solve(nlls, GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8)
    @test SciMLBase.successful_retcode(sol)
    res = estimate_parameters(pe, sol)
    @test isapprox(res.x[1], p_true; atol = 0.05)
    @test res.n_observations == length(t1) + length(t2)
    lo, hi = res.confidence_intervals[1]
    @test lo <= p_true <= hi

    # one fitted trajectory per experiment
    fp = fitted_parameters(pe, sol)
    @test fp.trajectory isa Tuple
    @test length(fp.trajectory) == 2

    # residuals are the concatenation over the experiments
    r = residual(pe, ps, st)
    @test length(r) == n_observations(data)
end

@testset "Wrong initial conditions bias the fit" begin
    # Experiment 2 falls back to the problem's initial conditions [1.0], which are wrong
    # for data generated from u0 = 2.0 -> the joint estimate is biased away from the truth.
    data_wrong = ExperimentData(ExperimentData(t1, y1; σ = σ), ExperimentData(t2, y2; σ = σ))
    pe = ParameterEstimationLayer(prob, Tsit5(), data_wrong)
    sol = solve(problem(pe), GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8)
    res = estimate_parameters(pe, sol)
    @test !isapprox(res.x[1], p_true; atol = 0.02)
end

@testset "Validation" begin
    # mixed noise scales are rejected
    data_mixed = ExperimentData(ExperimentData(t1, y1; σ = σ), ExperimentData(t2, y2))
    @test_throws ArgumentError ParameterEstimationLayer(prob, Tsit5(), data_mixed)
    # initial conditions with the wrong state dimension are rejected
    data_bad = ExperimentData(
        ExperimentData(t1, y1; σ = σ), ExperimentData(t2, y2; σ = σ, u0 = [1.0, 2.0]))
    @test_throws ArgumentError ParameterEstimationLayer(prob, Tsit5(), data_bad)
    # wrapping a single pre-built shooting layer keeps working for a single experiment
    layer = SingleShootingLayer(prob, Tsit5(); saveat = unique(t1))
    data_single = ExperimentData(t1, y1; σ = σ)
    pe = ParameterEstimationLayer(layer, data_single)
    @test pe isa ParameterEstimationLayer
    # the number of shooting layers must match the number of experiments
    @test_throws ArgumentError ParameterEstimationLayer((layer, layer), data_single)
end
