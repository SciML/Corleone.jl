using Test, CorleoneInverse, OrdinaryDiffEqTsit5, NonlinearSolve
using Optimization, OptimizationOptimJL, LikelihoodProfiler
using TransformVariables: asℝ₊
using Distributions: Chisq, quantile

t = collect(0.1:0.1:2.0)
ode = ODEProblem((u, p, t) -> [p[1] * t + p[2]], [1.0], (0.0, 2.0), [0.5, 0.2]; abstol = 1.0e-11, reltol = 1.0e-11)
measurements = 1 .+ 0.7 .* t .^ 2 ./ 2 .+ 0.3 .* t
data = Experiment(1 => (t, measurements, 0.1))

# Eliminate the nuisance control from the weighted linear design matrix.
function linear_profile_precision(t, noise_scale)
    rate, control = t .^ 2 / 2, t
    return (sum(abs2, rate) - sum(rate .* control)^2 / sum(abs2, control)) / noise_scale^2
end

@testset "Missing control metadata requires explicit targeting" begin
    problem = OptimizationProblem(
        OptimizationFunction((q, p) -> sum(abs2, q) / 2, AutoForwardDiff()), zeros(2);
        lb = fill(-1.0, 2), ub = fill(1.0, 2)
    )
    estimate = ParameterEstimate(
        zeros(2), [1.0 0.0; 0.0 1.0], ones(2), [(-1.0, 1.0), (-1.0, 1.0)],
        0.95, 0.0, 1, zeros(2), nothing
    )
    @test_throws ArgumentError ProfileLikelihoodProblem(problem, estimate)

    all_profile = ProfileLikelihoodProblem(problem, estimate; include_controls = true)
    @test all_profile.target.idxs == [1, 2]
    @test all_profile.optpars == estimate.internal_x

    selected_profile = ProfileLikelihoodProblem(problem, estimate; idxs = 2)
    @test selected_profile.target.idxs == [2]
    @test selected_profile.optpars == estimate.internal_x
end

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
    precision = linear_profile_precision(t, 0.1)
    for (x, objective) in zip(curve.x, curve.obj)
        @test objective ≈ precision * (x - 0.7)^2 / 2 atol = 1.0e-6
    end
    @test length(curve.x) > 2

    wrong = OptimizationProblem(OptimizationFunction((q, p) -> sum(abs2, q), AutoForwardDiff()), zeros(3))
    @test_throws DimensionMismatch ProfileLikelihoodProblem(wrong, estimate)
    standalone = ParameterEstimate(estimate.x, estimate.covariance, estimate.std_errors, estimate.confidence_intervals, estimate.level, estimate.rss, estimate.dof)
    @test_throws ArgumentError ProfileLikelihoodProblem(problem, standalone)
end

@testset "Float32 confidence levels cross the dependency boundary" begin
    shooting = ShootingLayer(ode, Tsit5(); controls = [2])
    model = CorleoneDynamicOptProblem(shooting, data)
    estimate = ParameterEstimate(model, [0.7, 0.3]; α = 0.05f0)
    @test estimate.level isa Float32
    problem = OptimizationProblem(model, AutoForwardDiff())
    profile = ProfileLikelihoodProblem(problem, estimate; profile_lower = -1.0, profile_upper = 2.0)
    @test profile.threshold ≈ quantile(Chisq(1), Float64(estimate.level))
    explicit = ProfileLikelihoodProblem(problem, estimate; conf_level = 0.9f0, profile_lower = -1.0, profile_upper = 2.0)
    @test explicit.threshold ≈ quantile(Chisq(1), Float64(0.9f0))
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

    bounded = OptimizationProblem(model, AutoForwardDiff(); lb = [log(0.2), -0.5], ub = [log(1.5), 1.0])
    bounded_profile = ProfileLikelihoodProblem(bounded, estimate; profile_lower = log(0.3), profile_upper = log(1.0))
    @test bounded_profile.target.profile_lower == [log(0.3)]
    @test bounded_profile.target.profile_upper == [log(1.0)]
    @test_throws ArgumentError ProfileLikelihoodProblem(bounded, estimate; profile_lower = log(0.1), profile_upper = log(1.0))
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

@testset "General constraints are rejected" begin
    shooting = ShootingLayer(ode, Tsit5(); controls = [2])
    for (lower, upper) in ((1.65, 1.65), (-Inf, 2.0))
        model = CorleoneDynamicOptProblem(shooting, data, :(u₁(1.0)); lcons = [lower], ucons = [upper], reducer = x -> sum(x) / 2)
        problem = OptimizationProblem(model, AutoForwardDiff())
        estimate = ParameterEstimate(model, [0.7, 0.3])
        @test_throws ArgumentError ProfileLikelihoodProblem(problem, estimate; profile_lower = -1.0, profile_upper = 2.0)
    end
end

@testset "Profile ranges respect optimization bounds" begin
    noise_scale = 0.025
    observations = Experiment(1 => (t, measurements, noise_scale))
    shooting = ShootingLayer(remake(ode; p = [0.7, 0.3]), Tsit5(); controls = [2])
    model = CorleoneDynamicOptProblem(shooting, observations; reducer = x -> sum(x) / 2)
    problem = OptimizationProblem(model, AutoForwardDiff(); lb = [0.6, -0.5], ub = [0.75, 1.0])
    estimate = ParameterEstimate(model, [0.7, 0.3])
    @test_throws ArgumentError ProfileLikelihoodProblem(problem, estimate; profile_lower = 0.4, profile_upper = 1.0)
    @test_throws ArgumentError ProfileLikelihoodProblem(problem, estimate; profile_lower = 0.4, profile_upper = 0.75)
    @test_throws ArgumentError ProfileLikelihoodProblem(problem, estimate; profile_lower = 0.6, profile_upper = 1.0)
    @test_throws ArgumentError ProfileLikelihoodProblem(problem, estimate; idxs = 2, profile_lower = -1.0, profile_upper = 1.0)
    @test_throws ArgumentError ProfileLikelihoodProblem(
        problem, estimate; include_controls = true,
        profile_lower = [0.4, -0.5], profile_upper = [1.0, 1.0]
    )

    cutoff = quantile(Chisq(1), estimate.level)
    profile = ProfileLikelihoodProblem(problem, estimate; threshold = cutoff / 2)
    @test profile.target.profile_lower == [0.6]
    @test profile.target.profile_upper == [0.75]
    # Resolve the interior cutoff crossings before a step reaches the box edge.
    profiles = solve(
        profile, OptimizationProfiler(;
            optimizer = LBFGS(), stepper = FixedStep(initial_step = 0.002),
            optimizer_opts = (; abstol = 1.0e-10)
        ); maxiters = 100
    )
    curve = only(profiles.profiles)
    @test all(x -> 0.6 <= x <= 0.75, curve.x)
    @test all(q -> all(problem.lb .<= q .<= problem.ub), curve.pars)
    interval = only(LikelihoodProfiler.endpoints(profiles))
    @test !isnothing(interval.left) && !isnothing(interval.right)
    if !isnothing(interval.left) && !isnothing(interval.right)
        @test 0.6 <= interval.left < estimate.x[1] < interval.right <= 0.75
        halfwidth = sqrt(cutoff / linear_profile_precision(t, noise_scale))
        @test interval.left ≈ 0.7 - halfwidth atol = 1.0e-4
        @test interval.right ≈ 0.7 + halfwidth atol = 1.0e-4
    end

    unbounded = OptimizationProblem(problem.f, problem.u0, problem.p)
    wide_profile = ProfileLikelihoodProblem(unbounded, estimate; profile_lower = 0.4, profile_upper = 1.0)
    @test wide_profile.target.profile_lower == [0.4]
    @test wide_profile.target.profile_upper == [1.0]
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
