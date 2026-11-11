using Test, CorleoneInverse, OrdinaryDiffEqTsit5, ComponentArrays, NonlinearSolve, LuxCore, StableRNGs
using Optimization, OptimizationMOI, Ipopt
using Distributions: LogNormal, Normal
using TransformVariables: asℝ₊

# u(t) = u₀ + a t with a Normal prior on a is a generalized ridge regression.
problem = ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 4.0), [0.5]; abstol = 1.0e-12, reltol = 1.0e-12)
t = collect(0.0:0.5:4.0)
noise = [0.3, -0.5, 0.2, 0.1, -0.4, 0.6, -0.2, 0.0, 0.1] ./ 10
y = 1.0 .+ 2.0 .* t .+ noise
σ, μ, τ = 0.1, 1.5, 0.05
experiment = Experiment(:u₁ => (t, y, σ))
X = [ones(length(t)) t]
# Unknowns are packed as [u₀, a]; the prior adds the row (a - μ) / τ.
P = [0.0 0.0; 0.0 1 / τ^2]
precision = X' * X / σ^2 + P
β = precision \ (X' * y / σ^2 + P * [0.0, μ])

function prior_layer(prior; transform = nothing)
    return ShootingLayer(
        problem, Tsit5(),
        Tunable(1; kind = Val(:parameter), default = 1.0, prior, transform) => nothing,
        Tunable(1; kind = Val(:variable)) => [0.0]
    )
end

@testset "Normal prior rows in least squares" begin
    layer = prior_layer(Normal(μ, τ))
    ps, st = LuxCore.setup(StableRNG(1), layer)
    nlls = NonlinearLeastSquaresProblem(layer, experiment, ps, st)
    @test length(nlls.f.resid_prototype) == length(t) + 1
    sol = solve(nlls, LevenbergMarquardt())
    @test sol.u ≈ β
    est = ParameterEstimate(layer, experiment, sol.u, ps, st)
    @test est.covariance ≈ inv(precision)
    @test est.rss ≈ sum(abs2, (y - X * β) / σ)
end

@testset "MAP through the optimization problem" begin
    layer = prior_layer(Normal(μ, τ))
    ps, st = LuxCore.setup(StableRNG(1), layer)
    optprob = OptimizationProblem(layer, experiment, ps, st, Optimization.AutoForwardDiff(); reducer = x -> sum(x) / 2)
    sol = solve(optprob, Ipopt.Optimizer(); print_level = 0, tol = 1.0e-12)
    @test sol.u ≈ β rtol = 1.0e-6
end

@testset "Unsupported least-squares priors" begin
    for layer in (prior_layer(LogNormal(0.0, 1.0)), prior_layer(Normal(μ, τ); transform = asℝ₊))
        ps, st = LuxCore.setup(StableRNG(1), layer)
        @test_throws ArgumentError NonlinearLeastSquaresProblem(layer, experiment, ps, st)
        @test OptimizationProblem(layer, experiment, ps, st, Optimization.AutoForwardDiff()) isa OptimizationProblem
    end
end

@testset "Prior statistics require known noise scales" begin
    layer = prior_layer(Normal(μ, τ))
    unweighted = Experiment(1 => (t, y))
    ps, st = LuxCore.setup(StableRNG(1), layer)
    model = CorleoneDynamicOptProblem(layer, unweighted; ps, st)
    q, _ = pack(default_vectorizer(), ps)
    @test_throws ArgumentError ParameterEstimate(model, q)
end

@testset "Least squares respects parameter bounds" begin
    # The unbounded slope is about 2; the bound keeps the fit at its upper limit.
    layer = ShootingLayer(
        problem, Tsit5(),
        Tunable(1; kind = Val(:parameter), default = 1.0, lowerbound = 0.0, upperbound = 1.5) => nothing,
        Tunable(1; kind = Val(:variable)) => [0.0]
    )
    ps, st = LuxCore.setup(StableRNG(1), layer)
    nlls = NonlinearLeastSquaresProblem(layer, experiment, ps, st)
    @test nlls.lb == [-Inf, 0.0]
    @test nlls.ub == [Inf, 1.5]
    sol = solve(nlls, LevenbergMarquardt())
    @test 1.4 < last(sol.u) <= 1.5
    unbounded = prior_layer(Normal(μ, τ))
    @test isnothing(NonlinearLeastSquaresProblem(unbounded, experiment, LuxCore.setup(StableRNG(1), unbounded)...).lb)
end
