using CorleoneInverse, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5, NonlinearSolve, StableRNGs, Test, LinearAlgebra
using Optimization, OptimizationMOI, Ipopt

@testset "Prepared symbolic least squares and uncertainty" begin
    problem = ODEProblem((u, p, t) -> p, [0.0], (0.0, 2.0), [1.0]; abstol = 1.0e-11, reltol = 1.0e-11)
    shooting = ShootingLayer(problem, Tsit5(), Tunable(1) => nothing, Tunable(1; kind = Val(:variable)) => [0.0])
    times = collect(0.0:0.2:2.0)
    X = hcat(fill(2.0, length(times)), 3 .* times)
    y = X * [1.0, 2.0] .+ 0.02 .* sin.(1:length(times))
    data = Experiment(:(2u₁ + p₁ * t) => (times, y, 0.1))
    model = CorleoneDynamicOptProblem(shooting, data)
    expected = X \ y
    fit = solve(NonlinearLeastSquaresProblem(model), LevenbergMarquardt())
    @test fit.u ≈ expected
    estimate = ParameterEstimate(model, fit.u)
    @test estimate.x ≈ expected
    @test estimate.covariance ≈ inv(X' * X / 0.1^2)
    @test estimate.dof == length(times) - 2
    @test estimate.rss ≈ sum(abs2, (y - X * expected) ./ 0.1)
    swapped_ps = (; parameter = model.ps.parameter, initialcondition = model.ps.initialcondition)
    swapped = CorleoneDynamicOptProblem(shooting, data; ps = swapped_ps, st = model.st)
    swapped_fit = solve(NonlinearLeastSquaresProblem(swapped), LevenbergMarquardt())
    swapped_estimate = ParameterEstimate(swapped, swapped_fit.u)
    @test swapped_estimate.x ≈ reverse(expected)
    @test swapped_estimate.covariance ≈ estimate.covariance[[2, 1], [2, 1]]

    opt = OptimizationProblem(model, Optimization.AutoForwardDiff())
    opt_fit = solve(opt, Ipopt.Optimizer(); print_level = 0, tol = 1.0e-10)
    @test opt_fit.u ≈ fit.u rtol = 1.0e-6
    @test opt.f.f(fit.u, opt.p) ≈ estimate.rss / length(times)

    unbounded = CorleoneDynamicOptProblem(shooting, data, :(u₁(0.3)); ps = model.ps, st = model.st, lcons = [-Inf], ucons = [Inf])
    @test NonlinearLeastSquaresProblem(unbounded).u0 == NonlinearLeastSquaresProblem(model).u0

    constrained = CorleoneDynamicOptProblem(shooting, data, :(u₁(0.0)); ps = model.ps, st = model.st, lcons = [1.0], ucons = [1.0])
    @test_throws ArgumentError NonlinearLeastSquaresProblem(constrained)
    constrained_opt = OptimizationProblem(constrained, Optimization.AutoForwardDiff())
    result = solve(constrained_opt, Ipopt.Optimizer(); print_level = 0, tol = 1.0e-10)
    rate = dot(X[:, 2], y .- 2) / dot(X[:, 2], X[:, 2])
    @test result.u ≈ [1.0, rate]
    conditional = ParameterEstimate(constrained, result.u)
    @test conditional.covariance[1, 1] ≈ 0.0 atol = 1.0e-14
    @test conditional.covariance[2, 2] ≈ 0.1^2 / sum(abs2, X[:, 2])
    @test conditional.dof == length(times) - 1
end

@testset "Solver parameters carry the current experiment state" begin
    problem = ODEProblem((u, p, t) -> [p[1] + p[2]], [0.0], (0.0, 1.0), [1.0, 0.0])
    shooting = ShootingLayer(problem, Tsit5(); controls = [2])
    data = Experiment(1 => ([1.0], [2.0]))
    experiments = Experiments(shooting, (run = (annotations = Constant(2) => nothing, observations = data),))
    model = CorleoneDynamicOptProblem(shooting, experiments)
    replacement = deepcopy(model.st)
    replacement.experiments.run.parameter.parameters.p₂.parameters.parameter.ps .= 0.5

    nlls = NonlinearLeastSquaresProblem(model)
    @test nlls.p === model.st
    @test nlls.f(nlls.u0, nlls.p) ≈ [1.0]
    @test nlls.f(nlls.u0, nothing) ≈ [1.0]
    updated = remake(nlls; p = replacement)
    @test updated.p === replacement
    @test updated.f(updated.u0, updated.p) ≈ [0.5]
    @test solve(updated, LevenbergMarquardt()).u ≈ [1.5]

    opt = OptimizationProblem(model, Optimization.AutoForwardDiff())
    @test opt.p === model.st
    @test opt.f.f(opt.u0, replacement) ≈ 0.25
end

@testset "Append experiments with a fitted warm start" begin
    problem = ODEProblem((u, p, t) -> [p[1] + p[2]], [0.0], (0.0, 1.0), [1.0, 0.0])
    shooting = ShootingLayer(problem, Tsit5(); controls = [2])
    a = (
        annotations = Tunable(2; default = [0.5, 1.0]),
        observations = Experiment(1 => ([0.5], [2.0])),
    )
    b = (
        annotations = Constant(2; default = 0.0) => nothing,
        observations = Experiment(1 => ([0.25, 0.75], [1.0, 3.0])),
    )
    batch = Experiments(shooting, (; a))
    model = CorleoneDynamicOptProblem(shooting, batch)
    ls = NonlinearLeastSquaresProblem(model)
    q = [2.0, 3.0, 4.0]
    appended = append_experiment(model, ls, :b => b; u0 = q)
    @test appended.problem.p === appended.model.st
    @test appended.problem.u0 == [2.0, 3.0, 3.0, 4.0, 4.0]
    @test appended.problem.p.saveat == [0.25, 0.5, 0.75]
    @test length(appended.problem.f.resid_prototype) == 3
    @test (@inferred appended.problem.f(appended.problem.u0, appended.problem.p)) ≈ [-0.5, 0.5, 1.5]
    @test ls.u0 == [1.0, 0.5, 1.0]
    @test_throws ArgumentError append_experiment(model, ls, :a => b)
    state = deepcopy(appended.problem.p)
    state.experiments.b.parameter.parameters.p₂.parameters.parameter.ps .= 1.0
    c = (
        annotations = Constant(2) => nothing,
        observations = Experiment(1 => ([0.25], [0.0])),
    )
    repeated = append_experiment(appended.model, appended.problem, :c => c; st = state)
    @test repeated.problem.u0 == appended.problem.u0
    @test repeated.problem.f(repeated.problem.u0, repeated.problem.p) ≈ [-0.5, 0.25, 0.75, -0.5]

    constrained = CorleoneDynamicOptProblem(shooting, batch, :(u₁(0.5)); lcons = [0.0], ucons = [Inf])
    opt = OptimizationProblem(constrained, Optimization.AutoForwardDiff())
    appended_opt = append_experiment(constrained, opt, :b => b; u0 = q)
    next = appended_opt.problem
    @test next.p === appended_opt.model.st
    @test next.u0 == appended.problem.u0
    @test (@inferred next.f.f(next.u0, next.p)) ≈ (0.25 + 0.25 + 2.25) / 3
    constraint = zeros(1)
    next.f.cons(constraint, next.u0, next.p)
    @test constraint ≈ [2.5]
end
