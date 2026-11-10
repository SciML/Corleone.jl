using Test, CorleoneInverse, LuxCore, Random, ComponentArrays, OrdinaryDiffEqTsit5, NonlinearSolve, LinearAlgebra
using Distributions: Normal, LogNormal
using TransformVariables: asℝ₊
using Optimization, OptimizationMOI, Ipopt, ForwardDiff

@testset "Joint experiment layer fit and covariance" begin
    problem = ODEProblem((u, p, t) -> [p[1] * t + p[2]], [1.0], (0.0, 2.0), [0.5, 0.2]; abstol = 1.0e-11, reltol = 1.0e-11)
    base = ShootingLayer(problem, Tsit5(), Tunable(1; kind = Val(:variable)) => [0.0]; controls = [2])
    t, σ = collect(0.1:0.1:2.0), 0.05
    # Packed order: shared coefficient, then initial value/control for a and b.
    X = vcat(
        hcat(t .^ 2 / 2, ones(length(t)), t, zeros(length(t)), zeros(length(t))),
        hcat(t .^ 2 / 2, zeros(length(t)), zeros(length(t)), ones(length(t)), t)
    )
    y = X * [0.7, 1.0, 0.2, 2.0, 0.4] + 0.002 .* cos.(1:size(X, 1))
    a = (annotations = (Tunable(1; kind = Val(:variable)) => [0.0], Tunable(2) => nothing), observations = (1 => (t, y[1:length(t)], σ),))
    b = (
        annotations = (Tunable(1; kind = Val(:variable), default = 1.8) => [0.0], Tunable(2) => nothing),
        observations = Experiment(1 => (t, y[(length(t) + 1):end], σ)),
    )
    batch = Experiments(base, (a = a, b = b))
    ps, st = LuxCore.setup(Xoshiro(1), batch)
    ls = NonlinearLeastSquaresProblem(base, batch, ps, st)
    fit = solve(ls, LevenbergMarquardt())
    estimate = ParameterEstimate(base, batch, fit.u, ps, st)
    @test estimate.x ≈ X \ y rtol = 1.0e-5
    @test estimate.covariance ≈ σ^2 * inv(X' * X) rtol = 1.0e-5
    @test estimate.dof == length(y) - 5
    @test abs(estimate.covariance[1, 2]) > 0
    @test abs(estimate.covariance[2, 4]) > 0
    packed = ComponentArray(ps)
    @test NonlinearLeastSquaresProblem(base, batch, packed, st).f(ls.u0, nothing) ≈ ls.f(ls.u0, nothing)
    @test ParameterEstimate(base, batch, fit.u, packed, st).covariance ≈ estimate.covariance
    no_data = Experiments(base, (a = (;),))
    @test_throws ArgumentError NonlinearLeastSquaresProblem(base, no_data, LuxCore.setup(Xoshiro(1), no_data)...)
end

@testset "Mixed fixed conditions and local transforms/priors" begin
    problem = ODEProblem((u, p, t) -> zero(u), [1.0], (0.0, 1.0), Float64[])
    base = ShootingLayer(problem, Tsit5())
    t, σ, τ = collect(0.1:0.1:1.0), 0.1, 0.4
    prior = LogNormal(0.0, τ)
    a = (
        annotations = (Tunable(1; kind = Val(:variable), default = 1.5, transform = asℝ₊, prior) => [0.0],),
        observations = (1 => (t, fill(2.0, length(t)), σ),),
    )
    b = (
        annotations = (Constant(1; kind = Val(:variable), default = 3.0) => [0.0],),
        observations = (1 => (t, fill(3.0, length(t)), σ),),
    )
    batch = Experiments(base, (a = a, b = b))
    ps, st = LuxCore.setup(Xoshiro(1), batch)
    q, _ = pack(default_vectorizer(), ps)
    @test length(q) == 1
    @test q ≈ [log(1.5)]
    @test isempty(ComponentArray(ps.shared))
    @test isempty(ComponentArray(ps.experiments.b))
    @test_throws ArgumentError NonlinearLeastSquaresProblem(base, batch, ps, st)
    opt = OptimizationProblem(base, batch, ps, st, Optimization.AutoForwardDiff())
    @test opt.lb == [-Inf]
    @test opt.ub == [Inf]
    expected = length(t) / σ^2 * (exp(q[1]) - 2.0) * exp(q[1]) + q[1] / τ^2
    @test only(ForwardDiff.gradient(x -> opt.f.f(x, nothing), q)) ≈ expected
    fit = solve(opt, Ipopt.Optimizer(); print_level = 0, tol = 1.0e-11)
    estimate = ParameterEstimate(base, batch, fit.u, ps, st)
    θ = only(estimate.x)
    @test only(estimate.covariance) ≈ θ^2 / (length(t) * θ^2 / σ^2 + 1 / τ^2) rtol = 1.0e-7
    @test estimate.dof == 2length(t) - 1
end

@testset "Different local shooting grids" begin
    problem = ODEProblem((u, p, t) -> [-p[1] * u[1] + p[2]], [1.0], (0.0, 2.0), [0.5, 0.0]; abstol = 1.0e-10, reltol = 1.0e-10)
    base = ShootingLayer(
        problem, Tsit5(), Tunable(1; kind = Val(:variable)) => [0.0],
        Constant(2) => nothing; controls = [2]
    )
    t, σ = collect(0.1:0.1:2.0), 0.03
    truth(t, initial, input) = (initial - input / 0.7) * exp(-0.7t) + input / 0.7
    data = (a = Experiment(1 => (t, truth.(t, 1.0, 0.0), σ)), b = Experiment(1 => (t, truth.(t, 2.0, 0.3), σ)))
    single = Experiments(
        base, (
            a = (annotations = (Tunable(1; kind = Val(:variable)) => [0.0],), observations = data.a),
            b = (annotations = (Tunable(1; kind = Val(:variable)) => [0.0], Constant(2; default = 0.3) => nothing), observations = data.b),
        )
    )
    p0, s0 = LuxCore.setup(Xoshiro(1), single)
    reference = solve(NonlinearLeastSquaresProblem(base, single, p0, s0), LevenbergMarquardt())
    reference_estimate = ParameterEstimate(base, single, reference.u, p0, s0)
    multiple = Experiments(
        base, (
            a = (annotations = (Tunable(1; kind = Val(:variable), default = [1.0, 0.5]) => [0.0, 1.0],), observations = data.a),
            b = (
                annotations = (
                    Tunable(1; kind = Val(:variable), default = [2.0, 1.3]) => [0.0, 0.7],
                    Constant(2; default = 0.3) => nothing,
                ), observations = data.b,
            ),
        )
    )
    ps, st = LuxCore.setup(Xoshiro(1), multiple)
    @test get_number_of_shooting_constraints(multiple, ps, st) == 2
    @test_throws ArgumentError NonlinearLeastSquaresProblem(base, multiple, ps, st)
    opt = OptimizationProblem(base, multiple, ps, st, Optimization.AutoForwardDiff())
    @test opt.lcons == opt.ucons == zeros(2)
    fit = solve(opt, Ipopt.Optimizer(); print_level = 0, tol = 1.0e-10, hessian_approximation = "limited-memory")
    estimate = ParameterEstimate(base, multiple, fit.u, ps, st)
    @test estimate.x[[1, 2, 4]] ≈ reference_estimate.x rtol = 1.0e-5
    @test estimate.covariance[[1, 2, 4], [1, 2, 4]] ≈ reference_estimate.covariance rtol = 1.0e-4
    @test estimate.dof == reference_estimate.dof
    @test estimate.rss < 1.0e-12
end

@testset "Global initial variables and controls are shared in the fit" begin
    τ, σ = 0.3, 0.05
    problem = ODEProblem((u, p, t) -> [p[1] * t + p[2]], [1.0], (0.0, 2.0), [0.5, 0.2]; abstol = 1.0e-11, reltol = 1.0e-11)
    base = ShootingLayer(problem, Tsit5(), Tunable(1; kind = Val(:variable), prior = Normal(1.0, τ)) => [0.0]; controls = [2])
    t = collect(0.1:0.1:2.0)
    X = vcat(
        hcat(ones(length(t)), t .^ 2 / 2, t, zeros(length(t))),
        hcat(ones(length(t)), t .^ 2 / 2, zeros(length(t)), t)
    )
    y = X * [1.0, 0.7, 0.2, 0.4] + 0.002 .* cos.(1:size(X, 1))
    batch = Experiments(
        base, (
            a = (observations = (1 => (t, y[1:length(t)], σ),),),
            b = (annotations = (Tunable(2; default = 0.3) => nothing,), observations = (1 => (t, y[(length(t) + 1):end], σ),)),
        )
    )
    ps, st = LuxCore.setup(Xoshiro(1), batch)
    @test length(ComponentArray(ps.shared)) == 3
    @test length(ComponentArray(ps.experiments)) == 1
    precision = X' * X / σ^2 + Diagonal([1 / τ^2, 0.0, 0.0, 0.0])
    expected = precision \ (X' * y / σ^2 + [1 / τ^2, 0.0, 0.0, 0.0])
    fit = solve(NonlinearLeastSquaresProblem(base, batch, ps, st), LevenbergMarquardt())
    estimate = ParameterEstimate(base, batch, fit.u, ps, st)
    @test estimate.x ≈ expected rtol = 1.0e-5
    @test estimate.covariance ≈ inv(precision) rtol = 1.0e-5
    @test estimate.dof == length(y) - 4
end

@testset "Individual ordinary parameters with shared coefficients" begin
    problem = ODEProblem((u, p, t) -> [p[1] * t + p[2]], [1.0], (0.0, 2.0), [0.5, 0.1]; abstol = 1.0e-11, reltol = 1.0e-11)
    shooting = ShootingLayer(problem, Tsit5())
    t, σ = collect(0.1:0.1:2.0), 0.05
    X = vcat(hcat(t, t .^ 2 / 2, zeros(length(t))), hcat(t, zeros(length(t)), t .^ 2 / 2))
    y = ones(2length(t)) + X * [0.2, 0.7, 1.1] + 0.002 .* cos.(1:size(X, 1))
    experiments = Experiments(
        problem; experiments = (
            a = (
                annotations = (Tunable(1; default = 0.6, prior = Normal(0.6, 0.2)) => nothing,),
                observations = (1 => (t, y[1:length(t)], σ),),
            ),
            b = (
                annotations = (Tunable(1; default = 1.0, prior = Normal(1.0, 0.3)) => nothing,),
                observations = (1 => (t, y[(length(t) + 1):end], σ),),
            ),
        )
    )
    ps, st = LuxCore.setup(Xoshiro(1), experiments)
    @test length(ComponentArray(ps.shared)) == 1
    @test length(ComponentArray(ps.experiments.a)) == 1
    @test length(ComponentArray(ps.experiments.b)) == 1
    precision = X' * X / σ^2 + Diagonal([0.0, 1 / 0.2^2, 1 / 0.3^2])
    expected = precision \ (X' * (y .- 1.0) / σ^2 + [0.0, 0.6 / 0.2^2, 1.0 / 0.3^2])
    fit = solve(NonlinearLeastSquaresProblem(shooting, experiments, ps, st), LevenbergMarquardt())
    estimate = ParameterEstimate(shooting, experiments, fit.u, ps, st)
    @test estimate.x ≈ expected rtol = 1.0e-5
    @test estimate.covariance ≈ inv(precision) rtol = 1.0e-5
    @test estimate.dof == length(y) - 3
    _, reconstruct = pack(default_vectorizer(), ps)
    trajectories, next = shooting(experiments, reconstruct(fit.u), st)
    @test trajectories isa Vector
    @test length(trajectories) == 2
    @test last(trajectories[1].u) != last(trajectories[2].u)
end
