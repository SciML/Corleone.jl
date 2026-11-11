using Test, Corleone, LuxCore, OrdinaryDiffEqTsit5, StableRNGs, ComponentArrays
using TransformVariables: asℝ₊, asℝ₋, as𝕀
using Distributions: Normal, LogNormal, logpdf
using SymbolicIndexingInterface: parameter_values
using ForwardDiff: derivative

problem = ODEProblem((u, p, t) -> [-p[1] * u[1], p[2] * u[1]], [1.0, 0.0], (0.0, 1.0), [0.5, 0.2])

@testset "Transformed parameters store internal values" begin
    layer = ShootingLayer(
        problem, Tsit5(),
        Tunable(1; kind = Val(:parameter), default = 2.0, lowerbound = 1.0, transform = asℝ₊) => nothing,
        Constant(2; kind = Val(:parameter), default = 0.25, transform = as𝕀) => nothing
    )
    ps, st = LuxCore.setup(StableRNG(1), layer)
    @test only(ComponentArray(ps)) ≈ log(2.0)
    traj, _ = layer(nothing, ps, st)
    @test parameter_values(traj) ≈ [2.0, 0.25]
    @test last(traj.u)[1] ≈ exp(-2.0) rtol = 1.0e-4
    lower, upper = get_bounds(layer, ps, st)
    @test only(ComponentArray(lower)) == -Inf
    @test only(ComponentArray(upper)) == Inf
    @test_throws ArgumentError ShootingLayer(problem, Tsit5(), Tunable(1; kind = Val(:parameter), transform = asℝ₋) => nothing)
end

@testset "Grouped and initial-condition transforms" begin
    layer = ShootingLayer(
        problem, Tsit5(),
        Tunable([1, 2]; kind = Val(:parameter), default = [2.0, 0.5], transform = [asℝ₊, nothing]) => nothing,
        Tunable(1; kind = Val(:variable), default = 3.0, transform = asℝ₊) => [0.5]
    )
    ps, st = LuxCore.setup(StableRNG(1), layer)
    @test ComponentArray(ps) ≈ [log(3.0), log(2.0), 0.5]
    traj, _ = layer(nothing, ps, st)
    @test parameter_values(traj) ≈ [2.0, 0.5]
    reset = findlast(==(0.5), traj.t)
    @test traj.u[reset][1] ≈ 3.0
    ps = constant_initialization(StableRNG(1), layer, ps, st; u0 = [4.0, 0.0])
    @test first(ComponentArray(ps)) ≈ log(4.0)
end

@testset "Log prior in natural coordinates" begin
    prior = LogNormal(0.0, 0.5)
    layer = ShootingLayer(
        problem, Tsit5(),
        Tunable(1; kind = Val(:parameter), default = 2.0, transform = asℝ₊, prior) => nothing,
        Tunable(2; kind = Val(:parameter), default = 0.3, prior = Normal(0.2, 0.1)) => [0.5]
    )
    ps, st = LuxCore.setup(StableRNG(1), layer)
    # The log-Jacobian of exp is the internal value log(2).
    expected = logpdf(prior, 2.0) + log(2.0) + 2 * logpdf(Normal(0.2, 0.1), 0.3)
    @test log_prior(layer, ps, st) ≈ expected
    @test log_prior(layer, ComponentArray(ps), st) ≈ expected
    @test length(Corleone.parameter_collections(layer, ps, st)) == 2
    @test_throws ArgumentError ShootingLayer(problem, Tsit5(), Tunable(1; kind = Val(:parameter), prior = :invalid) => nothing)
end

@testset "Bounded identity implements the transform protocol" begin
    transform = BoundedTransform(0.0, 4.0)
    @test get_bounds(transform) == (0.0, 4.0)
    @test get_bounds(BoundedTransform()) == (-Inf, Inf)
    @test Corleone.natural_value(transform, 2.0) == 2.0
    @test Corleone.internal_value(transform, 2.0) == 2.0
    @test Corleone.log_jacobian(transform, 2.0) == 0.0
    @test derivative(x -> Corleone.natural_value(transform, x), 2.0) == 1.0
    values = [1.0, 2.0]
    @test Corleone.natural_values(transform, values) === values
    @test Corleone.internal_values(transform, values) === values
    annotation = Tunable(1; lowerbound = 0.0, upperbound = 4.0)
    @test annotation.transform isa BoundedTransform
    @test !hasfield(typeof(annotation), :lowerbound)
    @test !hasfield(typeof(annotation), :upperbound)
    @test get_bounds(annotation) == get_bounds(transform)
    @test get_bounds(Corleone.set_lowerbound(annotation, 1.0)) == (1.0, 4.0)
    replacement = BoundedTransform(-2.0, 3.0)
    @test Corleone.set_transform(annotation, replacement).transform === replacement
    grouped = Corleone.set_transform(Tunable([1, 2]), [nothing, asℝ₊])
    @test first(grouped.transform) isa BoundedTransform
    @test last(grouped.transform) === asℝ₊
end

@testset "Explicit transforms replace identity bounds" begin
    @test get_bounds(asℝ₊) == get_bounds(as𝕀) == (-Inf, Inf)
    annotation = Tunable(1; default = 0.5, lowerbound = 1.0, upperbound = 3.0, transform = asℝ₊)
    @test annotation.transform === asℝ₊
    @test get_bounds(annotation) == (-Inf, Inf)
    layer = ShootingLayer(problem, Tsit5(), annotation => nothing)
    ps, st = LuxCore.setup(StableRNG(1), layer)
    @test ps.parameter.parameters.p₁.parameter.p₁.ps ≈ [log(0.5)]
    lower, upper = get_bounds(layer, ps, st)
    @test all(==(-Inf), ComponentArray(lower))
    @test all(==(Inf), ComponentArray(upper))
    grouped = Tunable([1, 2]; lowerbound = [0.0, 0.2], upperbound = [4.0, 0.8], transform = [nothing, as𝕀])
    expanded = Corleone.scalar_annotations(grouped)
    @test get_bounds(first(expanded)) == (0.0, 4.0)
    @test last(expanded).transform === as𝕀
    @test_throws DimensionMismatch Corleone.scalar_annotations(Tunable([1, 2]; transform = [asℝ₊]))
end

@testset "Forward initialization preserves transformed coordinates" begin
    layer = ShootingLayer(
        problem, Tsit5(),
        Tunable(1; kind = Val(:variable), default = 2.0, transform = asℝ₊) => [0.0, 0.5]
    )
    ps, st = LuxCore.setup(StableRNG(1), layer)
    forward_initialization(StableRNG(1), layer, ps, st; variables = [1])
    values = ps.initialcondition.initialconditions.u₁.parameter.ps
    @test values ≈ [log(2.0), log(2.0) - 0.25] rtol = 1.0e-4
    traj, _ = layer(nothing, ps, st)
    @test maximum(abs, shooting_constraints(traj)) < 1.0e-5
end

@testset "Inserted shooting segments do not repeat priors" begin
    prior = Normal(0.4, 0.1)
    layer = ShootingLayer(
        problem, Tsit5(),
        Tunable([1, 2]; kind = Val(:variable), default = [1.0, 0.0]) => [0.0, 0.5],
        Tunable(1; kind = Val(:parameter), default = 0.5, prior) => nothing
    )
    ps, st = LuxCore.setup(StableRNG(1), layer)
    expected = log_prior(layer, ps, st)
    @test expected ≈ logpdf(prior, 0.5)
    expanded, state = add_parameter_segments!(layer, ps, st)
    @test expanded.parameter.parameters.p₁.parameter.p₁.ps == [0.5, 0.5]
    @test log_prior(layer, expanded, state) ≈ expected
    @test log_prior(layer, ComponentArray(expanded), state) ≈ expected
    prior_collections = filter(entry -> !isnothing(first(entry).prior), Corleone.parameter_collections(layer, expanded, state))
    @test last(only(prior_collections)) == [2]
end

@testset "Explicit transforms keep symbolic bounds metadata" begin
    layer = ShootingLayer(
        problem, Tsit5(), Tunable(1; kind = Val(:parameter), default = 2.0, transform = asℝ₊) => nothing;
        bounds = Dict(:p₁ => (1.0, 3.0))
    )
    @test get_bounds(layer.sys, :p₁) == (1.0, 3.0)
    ps, st = LuxCore.setup(StableRNG(1), layer)
    lower, upper = get_bounds(layer, ps, st)
    @test first(ComponentArray(lower)) == -Inf
    @test first(ComponentArray(upper)) == Inf
end

@testset "Values must lie in the transform domain" begin
    for (transform, value) in ((asℝ₊, 0.0), (asℝ₊, -1.0), (as𝕀, 1.0))
        @test_throws ArgumentError ShootingLayer(problem, Tsit5(), Tunable(1; kind = Val(:parameter), default = value, transform) => nothing)
    end
    # The model value 0.0 of u₂ is the endpoint of the positive domain.
    @test_throws ArgumentError ShootingLayer(problem, Tsit5(), Tunable(2; kind = Val(:variable), transform = asℝ₊) => [0.0])
    callback = Corleone.PiecewiseConstant(nothing, (rng, n) -> zeros(n); transform = asℝ₊)
    @test_throws ArgumentError LuxCore.setup(StableRNG(1), callback)
    layer = ShootingLayer(problem, Tsit5(), Tunable(1; kind = Val(:variable), default = 2.0, transform = asℝ₊) => [0.0, 0.5])
    ps, st = LuxCore.setup(StableRNG(1), layer)
    @test_throws ArgumentError constant_initialization(StableRNG(1), layer, ps, st; u0 = [0.0, 0.0])
end
