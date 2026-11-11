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
