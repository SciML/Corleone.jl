using Test, CorleoneInverse, OrdinaryDiffEqTsit5, NonlinearSolve, LuxCore, StableRNGs
using Distributions: Chisq, FDist, quantile
using TransformVariables: asℝ₊

# u(t) = u₀ + a t is linear in (u₀, a), so Beale's region is the ordinary
# least-squares ellipse and every residual sum of squares has a closed form.
problem = ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 4.0), [0.5]; abstol = 1.0e-12, reltol = 1.0e-12)
t = collect(0.0:0.5:4.0)
noise = [0.3, -0.5, 0.2, 0.1, -0.4, 0.6, -0.2, 0.0, 0.1] ./ 10
y = 1.0 .+ 2.0 .* t .+ noise
X = [ones(length(t)) t]
n, p = size(X)
β = X \ y
ssr(θ, y = y) = sum(abs2, y - X * θ)

linear_layer(; slope...) = ShootingLayer(
    problem, Tsit5(),
    Tunable(1; kind = Val(:parameter), merge((; default = 1.0), slope)...) => nothing,
    Tunable(1; kind = Val(:variable)) => [0.0]
)

function fit(layer, experiment, constraints...; kwargs...)
    ps, st = LuxCore.setup(StableRNG(1), layer)
    sol = solve(NonlinearLeastSquaresProblem(layer, experiment, ps, st), LevenbergMarquardt())
    model = CorleoneDynamicOptProblem(layer, experiment, constraints...; ps, st, kwargs...)
    return model, ParameterEstimate(model, sol.u)
end

# Fraction of a uniform sample of the box that the analytic SSR accepts.
function accepted_fraction(result, ssr; samples = 20_000)
    rng = StableRNG(99)
    return count(1:samples) do _
        θ = result.lower .+ (result.upper .- result.lower) .* rand(rng, length(result.lower))
        ssr(θ) < result.threshold
    end / samples
end

@testset "Unweighted region matches the least-squares F region" begin
    model, est = fit(linear_layer(), Experiment(:u₁ => (t, y)))
    @test est.x ≈ β
    result = beale_joint_confidence(model, est; names = (:u₀, :a), n_accept = 400, rng = StableRNG(2))
    @test result.names == ["u₀", "a"]
    @test result.estimate ≈ β
    @test result.minimal_ssr ≈ ssr(β)
    @test result.threshold ≈ ssr(β) * (1 + p / (n - p) * quantile(FDist(p, n - p), 0.95))
    @test size(result.parameters) == (2, 400)
    # The returned SSR values agree with the closed form, so only the ellipse is accepted.
    @test result.ssr ≈ map(ssr, eachcol(result.parameters))
    @test all(<(result.threshold), result.ssr)
    # Twice the marginal intervals reach beyond the ellipse: candidates are rejected
    # at the rate the closed form predicts.
    @test result.tested > 400
    @test 400 / result.tested ≈ accepted_fraction(result, ssr) rtol = 0.15
    # Without transforms the box doubles the marginal intervals around the estimate.
    halfwidths = map(((lo, hi),) -> hi - lo, est.confidence_intervals)
    @test result.upper .- result.lower ≈ 2 .* halfwidths
    @test (result.lower .+ result.upper) ./ 2 ≈ β
end

@testset "Known noise scales use the chi-squared region" begin
    model, est = fit(linear_layer(), Experiment(:u₁ => (t, y, 0.1)))
    weighted_ssr(θ) = ssr(θ) / 0.1^2
    result = beale_joint_confidence(model, est; n_accept = 100, rng = StableRNG(3))
    @test result.minimal_ssr ≈ weighted_ssr(β)
    @test result.threshold ≈ weighted_ssr(β) + quantile(Chisq(p), 0.95)
    @test result.ssr ≈ map(weighted_ssr, eachcol(result.parameters))
    @test 100 / result.tested ≈ accepted_fraction(result, weighted_ssr) rtol = 0.25
end

@testset "Transformed parameters are sampled inside their domain" begin
    # A slope of about 0.1 with an interval that reaches far beyond zero in
    # natural coordinates of an untransformed fit.
    noisy = 1.0 .+ 0.1 .* t .+ 20 .* noise
    model, est = fit(linear_layer(transform = asℝ₊), Experiment(:u₁ => (t, noisy)))
    result = beale_joint_confidence(model, est; expansion = 4.0, n_accept = 100, rng = StableRNG(24))
    @test size(result.parameters) == (2, 100)
    @test all(>(0), result.parameters[2, :])
    @test result.lower[2] > 0
    @test result.ssr ≈ map(θ -> ssr(θ, noisy), eachcol(result.parameters))
end

@testset "Samples respect parameter bounds and inequality constraints" begin
    bounded = linear_layer(default = β[2], lowerbound = β[2] - 0.01, upperbound = β[2] + 0.01)
    model, est = fit(bounded, Experiment(:u₁ => (t, y)))
    result = beale_joint_confidence(model, est; n_accept = 100, rng = StableRNG(4))
    @test all(a -> β[2] - 0.01 <= a <= β[2] + 0.01, result.parameters[2, :])
    @test result.lower[2] >= β[2] - 0.01
    @test result.upper[2] <= β[2] + 0.01

    # u(4) = u₀ + 4a ≤ limit is inactive at the estimate and cuts the region.
    limit = β[1] + 4 * β[2] + 0.02
    model, est = fit(linear_layer(), Experiment(:u₁ => (t, y)), :(u₁(4.0)); lcons = [-Inf], ucons = [limit])
    result = beale_joint_confidence(model, est; n_accept = 100, rng = StableRNG(5))
    @test all(θ -> θ[1] + 4 * θ[2] <= limit, eachcol(result.parameters))
    @test result.tested > 100
end

@testset "Model evaluations outside the model's domain are rejected" begin
    # √a fails for negative slopes, which the untransformed box includes.
    root = ODEProblem((u, p, t) -> [sqrt(p[1])], [0.0], (0.0, 4.0), [0.5]; abstol = 1.0e-12, reltol = 1.0e-12)
    layer = ShootingLayer(
        root, Tsit5(), Tunable(1; kind = Val(:parameter), default = 0.5) => nothing,
        Tunable(1; kind = Val(:variable)) => [0.0]
    )
    model, est = fit(layer, Experiment(:u₁ => (t, 1.0 .+ 0.1 .* t .+ 5 .* noise)))
    @test first(est.confidence_intervals[2]) < 0
    result = beale_joint_confidence(model, est; n_accept = 50, rng = StableRNG(6))
    @test all(>=(0), result.parameters[2, :])
end

@testset "Unsupported models and arguments" begin
    model, est = fit(linear_layer(), Experiment(:u₁ => (t, y)))
    @test_throws ArgumentError beale_joint_confidence(model, est; names = ["only one"])
    @test_throws ArgumentError beale_joint_confidence(model, est; confidence = 1)
    @test_throws ArgumentError beale_joint_confidence(model, est; expansion = 0)
    @test_throws ArgumentError beale_joint_confidence(model, est; n_accept = 0)

    # Equality constraints, including multiple-shooting matching, are not sampled.
    constrained, constrained_est = fit(
        linear_layer(), Experiment(:u₁ => (t, y)), :(u₁(0.0)); lcons = [β[1]], ucons = [β[1]]
    )
    @test_throws "equality constraints" beale_joint_confidence(constrained, constrained_est)
end
