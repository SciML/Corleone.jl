using Test, CorleoneInverse, OrdinaryDiffEqTsit5, NonlinearSolve
using Optimization, OptimizationOptimJL, LikelihoodProfiler
using TransformVariables: asℝ₊
using Distributions: Chisq, quantile

t = collect(0.1:0.1:2.0)
ode = ODEProblem((u, p, t) -> [p[1] * t + p[2]], [1.0], (0.0, 2.0), [0.5, 0.2]; abstol = 1.0e-11, reltol = 1.0e-11)
data = Experiment(1 => (t, 1 .+ 0.7 .* t .^ 2 ./ 2 .+ 0.3 .* t, 0.1))

@testset "Fitted profile and control selection" begin
    # Both parameters share a grid, but only the second is a control.
    shooting = ShootingLayer(ode, Tsit5(); controls = [2])
    model = CorleoneDynamicOptProblem(shooting, data; reducer = x -> sum(x) / 2)
    problem = OptimizationProblem(model, AutoForwardDiff())
    fit = solve(NonlinearLeastSquaresProblem(model), LevenbergMarquardt())
    estimate = ParameterEstimate(model, fit.u; α = 0.1)
    @test estimate.control_indices == [2]
    @test estimate.internal_x ≈ fit.u
    @test estimate.internal_x !== fit.u
    profile = ProfileLikelihoodProblem(problem, estimate; profile_lower = -1.0, profile_upper = 2.0)
    @test profile.optprob === problem
    @test profile.optpars ≈ fit.u
    @test profile.optpars !== estimate.internal_x
    @test profile.target.idxs == [1]
    @test profile.threshold ≈ quantile(Chisq(1), estimate.level)
    all_profile = ProfileLikelihoodProblem(problem, estimate; include_controls = true, profile_lower = -1.0, profile_upper = 2.0)
    @test all_profile.target.idxs == [1, 2]
    control_profile = ProfileLikelihoodProblem(problem, estimate; idxs = 2, profile_lower = -1.0, profile_upper = 2.0, conf_level = 0.95, threshold = 1.0)
    @test control_profile.target.idxs == [2]
    @test control_profile.threshold == 1.0

    # Actual nuisance-parameter reoptimization follows the analytical quadratic.
    profiled = solve(
        profile, OptimizationProfiler(;
            optimizer = LBFGS(), stepper = FixedStep(initial_step = 0.05),
            optimizer_opts = (; abstol = 1.0e-10)
        ); maxiters = 100
    )
    curve = only(profiled.profiles)
    for (x, objective) in zip(curve.x, curve.obj)
        @test objective ≈ problem.f(fit.u, problem.p) + (x - fit.u[1])^2 / (2estimate.covariance[1, 1]) atol = 1.0e-6
    end
    @test length(curve.x) > 2

    wrong = OptimizationProblem(OptimizationFunction((q, p) -> sum(abs2, q), AutoForwardDiff()), zeros(3))
    @test_throws DimensionMismatch ProfileLikelihoodProblem(wrong, estimate)
    standalone = ParameterEstimate(estimate.x, estimate.covariance, estimate.std_errors, estimate.confidence_intervals, estimate.level, estimate.rss, estimate.dof)
    @test_throws ArgumentError ProfileLikelihoodProblem(problem, standalone)
end

@testset "Transforms retain internal coordinates" begin
    shooting = ShootingLayer(ode, Tsit5(), Tunable(1; transform = asℝ₊) => nothing; controls = [2])
    model = CorleoneDynamicOptProblem(shooting, data; reducer = x -> sum(x) / 2)
    problem = OptimizationProblem(model, AutoForwardDiff())
    q = [log(0.7), 0.3]
    estimate = ParameterEstimate(model, q)
    profile = ProfileLikelihoodProblem(problem, estimate; profile_lower = -2.0, profile_upper = 1.0)
    @test estimate.x ≈ [0.7, 0.3]
    @test profile.optpars ≈ q
    @test profile.target.idxs == [1]
    @test profile.optprob.f(profile.optpars, problem.p) < 1.0e-12
end

@testset "Experiment-local and fixed controls" begin
    shooting = ShootingLayer(ode, Tsit5(); controls = [2])
    experiments = Experiments(
        shooting, (
            a = (annotations = (Tunable(2) => nothing,), observations = data),
            b = (annotations = (Constant(2; default = 0.3) => nothing,), observations = data),
        )
    )
    model = CorleoneDynamicOptProblem(shooting, experiments; reducer = x -> sum(x) / 2)
    problem = OptimizationProblem(model, AutoForwardDiff())
    estimate = ParameterEstimate(model, [0.7, 0.3])
    @test estimate.control_indices == [2]
    profile = ProfileLikelihoodProblem(problem, estimate; profile_lower = -1.0, profile_upper = 2.0)
    @test profile.target.idxs == [1]
    @test length(profile.optpars) == 2

    all_controls = ShootingLayer(ode, Tsit5(); controls = [1, 2])
    model = CorleoneDynamicOptProblem(all_controls, data)
    problem = OptimizationProblem(model, AutoForwardDiff())
    estimate = ParameterEstimate(model, [0.7, 0.3])
    @test_throws ArgumentError ProfileLikelihoodProblem(problem, estimate; profile_lower = -1.0, profile_upper = 2.0)
    @test ProfileLikelihoodProblem(problem, estimate; include_controls = true, profile_lower = -1.0, profile_upper = 2.0).target.idxs == [1, 2]
end

@testset "Constrained problem is retained" begin
    shooting = ShootingLayer(ode, Tsit5(); controls = [2])
    model = CorleoneDynamicOptProblem(shooting, data, :(u₁(1.0)); lcons = [1.65], ucons = [1.65], reducer = x -> sum(x) / 2)
    problem = OptimizationProblem(model, AutoForwardDiff())
    estimate = ParameterEstimate(model, [0.7, 0.3])
    profile = ProfileLikelihoodProblem(problem, estimate; profile_lower = -1.0, profile_upper = 2.0)
    @test profile.optprob === problem
    @test profile.optprob.f.cons === problem.f.cons
    @test profile.optprob.lcons == profile.optprob.ucons == [1.65]
end

@testset "Piecewise controls and finite optimization bounds" begin
    shooting = ShootingLayer(
        ode, Tsit5(),
        Tunable(1; lowerbound = 0.0, upperbound = 2.0) => nothing,
        Tunable(2; default = [0.2, 0.4]) => [1.0]; controls = [2]
    )
    model = CorleoneDynamicOptProblem(shooting, data; reducer = x -> sum(x) / 2)
    problem = OptimizationProblem(model, AutoForwardDiff())
    estimate = ParameterEstimate(model, [0.7, 0.3, 0.3])
    @test estimate.control_indices == [2, 3]
    profile = ProfileLikelihoodProblem(problem, estimate)
    @test profile.target.idxs == [1]
    @test profile.target.profile_lower == [0.0]
    @test profile.target.profile_upper == [2.0]
    @test profile.optprob.lb === problem.lb
    @test profile.optprob.ub === problem.ub
end

@testset "No controls and experiment-owned control metadata" begin
    shooting = ShootingLayer(ode, Tsit5())
    model = CorleoneDynamicOptProblem(shooting, data)
    estimate = ParameterEstimate(model, [0.7, 0.3])
    @test isempty(estimate.control_indices)
    profile = ProfileLikelihoodProblem(
        OptimizationProblem(model, AutoForwardDiff()), estimate;
        profile_lower = -1.0, profile_upper = 2.0
    )
    @test profile.target.idxs == [1, 2]

    experiments = Experiments(
        ode; controls = [1], experiments = (
            a = (annotations = (Tunable(1) => nothing,), observations = data),
            b = (annotations = (Tunable(1) => nothing,), observations = data),
        )
    )
    model = CorleoneDynamicOptProblem(shooting, experiments)
    estimate = ParameterEstimate(model, [0.3, 0.7, 0.7])
    @test estimate.control_indices == [2, 3]
    profile = ProfileLikelihoodProblem(
        OptimizationProblem(model, AutoForwardDiff()), estimate;
        profile_lower = -1.0, profile_upper = 2.0
    )
    @test profile.target.idxs == [1]
end
