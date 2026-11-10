using Test, CorleoneInverse, OrdinaryDiffEqTsit5, ComponentArrays, NonlinearSolve, LuxCore, StableRNGs
using TransformVariables: asℝ₊

function lotka(u, p, t)
    return [p[1] * u[1] - p[2] * u[1] * u[2], -p[3] * u[2] + p[2] * u[1] * u[2]]
end
θ = [1.5, 1.0, 3.0]
problem = ODEProblem(lotka, [1.0, 1.0], (0.0, 5.0), θ; abstol = 1.0e-10, reltol = 1.0e-10)
t = collect(0.25:0.25:5.0)
function measure(u0; kwargs...)
    Y = reduce(hcat, solve(remake(problem; u0), Tsit5(); saveat = t).u)
    return Experiment(:u₁ => (t, Y[1, :], 0.05), :u₂ => (t[1:2:end], Y[2, 1:2:end], 0.05); kwargs...)
end
experiments = (measure([1.0, 1.0]), measure([0.5, 2.0]; u0 = [0.5, 2.0]))
layer = ShootingLayer(
    problem, Tsit5(),
    Tunable(1; kind = Val(:parameter), default = 1.2) => nothing,
    Corleone.Constant(2; kind = Val(:parameter)) => nothing,
    Tunable(3; kind = Val(:parameter), default = 2.5) => nothing
)
ps, st = LuxCore.setup(StableRNG(1), layer)

sol = solve(NonlinearLeastSquaresProblem(layer, experiments, ps, st), LevenbergMarquardt())
est = ParameterEstimate(layer, experiments, sol.u, ps, st)


@testset "Shared parameters over two experiments" begin
    states = map(experiment -> experiment_state(layer, experiment, st), experiments)
    trajectories = map((experiment, state) -> first(layer(experiment, ps, state)), experiments, states)
    @test first(last(trajectories).u) == [0.5, 2.0]
    @test n_observations(experiments) == 2 * (length(t) + length(t[1:2:end]))
    @test length(residual(experiments, trajectories)) == n_observations(experiments)
    sol = solve(NonlinearLeastSquaresProblem(layer, experiments, ps, st), LevenbergMarquardt())
    @test sol.u ≈ θ[[1, 3]] rtol = 1.0e-6
    est = ParameterEstimate(layer, experiments, sol.u, ps, st)
    @test est.rss < 1.0e-10
    @test length(est.x) == 2
end

positive = ShootingLayer(
    problem, Tsit5(),
    Tunable(1; kind = Val(:parameter), default = 1.2, transform = asℝ₊) => nothing,
    Corleone.Constant(2; kind = Val(:parameter)) => nothing,
    Tunable(3; kind = Val(:parameter), default = 2.5, transform = asℝ₊) => nothing
)

@testset "Positive transforms on annotations" begin
    positive = ShootingLayer(
        problem, Tsit5(),
        Tunable(1; kind = Val(:parameter), default = 1.2, transform = asℝ₊) => nothing,
        Constant(2; kind = Val(:parameter)) => nothing,
        Tunable(3; kind = Val(:parameter), default = 2.5, transform = asℝ₊) => nothing
    )
    ps, st = LuxCore.setup(StableRNG(1), positive)
    nlls = NonlinearLeastSquaresProblem(positive, experiments, ps, st)
    @test nlls.u0 ≈ log.([1.2, 2.5])
    sol = solve(nlls, LevenbergMarquardt())
    @test exp.(sol.u) ≈ θ[[1, 3]] rtol = 1.0e-6
    est = ParameterEstimate(positive, experiments, sol.u, ps, st)
    @test est.x ≈ θ[[1, 3]] rtol = 1.0e-6
    @test all(map((ci, x) -> 0 < first(ci) < x < last(ci), est.confidence_intervals, est.x))
    # At the optimum, the delta method recovers the untransformed covariance.
    ps0, st0 = LuxCore.setup(StableRNG(1), layer)
    reference = ParameterEstimate(layer, experiments, est.x, ps0, st0)
    @test est.covariance ≈ reference.covariance rtol = 1.0e-6
end
