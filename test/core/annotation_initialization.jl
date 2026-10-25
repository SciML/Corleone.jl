using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test

struct BothInitializerSignatures end
(::BothInitializerSignatures)(rng, n::Int) = fill(7.0, n)
(::BothInitializerSignatures)(rng, times::AbstractVector) = fill(-7.0, length(times))

@testset "Annotation defaults on independent state grids" begin
    prob = ODEProblem((u, p, t) -> zero(u), [1.0, 2.0], (0.0, 4.0), [0.5])
    grid = [3.0, 1.0] # Values follow the supplied order, rather than sorted times.
    supplied = [30.0, 10.0]
    for constructor in (Tunable, Constant), default in (
            2.0, supplied, (rng, n::Int) -> fill(4.0, n),
            (rng, times::AbstractVector) -> 10 .* times,
            (rng, n) -> fill(5.0, n), BothInitializerSignatures())
        layer = InitialConditions(prob,
            constructor(1; default, lowerbound = 0.0, upperbound = 40.0) => grid)
        ps, st = LuxCore.setup(MersenneTwister(1), layer)
        values = constructor === Tunable ? ps.initialconditions.u₁.parameter.ps :
            st.initialconditions.u₁.parameters.parameter.ps
        expected = default isa Number ? fill(default, 2) : default === supplied ? supplied :
            default isa BothInitializerSignatures ? fill(7.0, 2) :
            applicable(default, MersenneTwister(1), 2) ? default(MersenneTwister(1), 2) : 10 .* grid
        @test values == expected
        default === supplied && @test values === supplied
        (u, residuals), _ = layer(([0.0, 2.0], 1.0), ps, st)
        @test u == [expected[2], 2.0]
        @test residuals == (constructor === Tunable ? [expected[2]] : Float64[])
    end

    calls = Ref(0)
    initialize = (rng, times::AbstractVector) -> begin
        calls[] += 1
        times .+ rand(rng, length(times))
    end
    layer = InitialConditions(prob, Tunable(1; default = initialize) => grid,
        Tunable(2; default = (rng, times::AbstractVector) -> times.^2) => [2.0])
    @test LuxCore.parameterlength(layer) == 3
    @test calls[] == 0
    ps, st = LuxCore.setup(MersenneTwister(42), layer)
    @test calls[] == 1
    @test ps.initialconditions.u₂.parameter.ps == [4.0]
    again, _ = LuxCore.setup(MersenneTwister(42), layer)
    @test calls[] == 2
    @test again == ps
    layer(([1.0, 2.0], 3.0), ps, st)
    @test calls[] == 2 # No initialization on the forward path.

    fixed_calls = Ref(0)
    fixed = InitialConditions(prob, Constant(1; default = (rng, times::AbstractVector) -> begin
        fixed_calls[] += 1
        ones(length(times))
    end) => grid)
    @test LuxCore.parameterlength(fixed) == 0
    @test LuxCore.statelength(fixed) > 0
    @test fixed_calls[] == 0
    LuxCore.setup(MersenneTwister(1), fixed)
    @test fixed_calls[] == 1

    for grid in (nothing, Float64[])
        layer = InitialConditions(prob,
            Tunable(1; default = (rng, times::AbstractVector) -> zero.(times)) => grid)
        ps, _ = LuxCore.setup(MersenneTwister(1), layer)
        @test isempty(ps.initialconditions.u₁.parameter.ps)
    end
end

@testset "Parameter defaults use interval start times" begin
    prob = ODEProblem((u, p, t) -> zero(u), [1.0], (2.0, 6.0), [0.5])
    for constructor in (Tunable, Constant), grid in (nothing, [3.0, 5.0])
        times = isnothing(grid) ? [2.0] : [2.0, 3.0, 5.0]
        for default in (times, (rng, times::AbstractVector) -> times,
                (rng, n::Int) -> collect(1.0:n), (rng, n) -> ones(n))
            layer = Parameter(prob, constructor(1; default) => grid)
            ps, st = LuxCore.setup(MersenneTwister(1), layer)
            stored = constructor === Tunable ? ps.parameters.p₁.parameter.ps :
                st.parameters.p₁.parameters.parameter.ps
            expected = default isa AbstractVector ? default :
                applicable(default, MersenneTwister(1), length(times)) ?
                    default(MersenneTwister(1), length(times)) : times
            @test stored == expected
            @test first(layer(2.0, ps, st)) == [first(expected)]
            if !isnothing(grid)
                @test first(layer(3.0, ps, st)) == [expected[2]]
                @test first(layer(5.0, ps, st)) == [expected[3]]
            end
        end
    end
end

@testset "Initializer validation and error propagation" begin
    prob = ODEProblem((u, p, t) -> zero(u), [1.0], (0.0, 4.0), [0.5])
    for constructor in (Tunable, Constant)
        @test_throws ArgumentError InitialConditions(prob,
            constructor(1; default = [-1.0, 1.0], lowerbound = 0.0) => [1.0, 2.0])
        @test_throws ArgumentError Parameter(prob,
            constructor(1; default = [0.0, 2.0], upperbound = 1.0) => [1.0])
        for default in ([1.0], (rng, n::Int) -> ones(n + 1),
                (rng, times::AbstractVector) -> ones(length(times) + 1))
            layer = InitialConditions(prob, constructor(1; default) => [1.0, 2.0])
            @test_throws DimensionMismatch LuxCore.setup(MersenneTwister(1), layer)
        end
        for default in ((rng, n::Int) -> fill(-1.0, n),
                (rng, times::AbstractVector) -> fill(2.0, length(times)),
                (rng, n) -> fill(NaN, n), (rng, n) -> ones(1, n),
                (rng, n) -> 0.0, (rng,) -> [0.0])
            layer = InitialConditions(prob,
                constructor(1; default, lowerbound = 0.0, upperbound = 1.0) => [1.0, 2.0])
            @test_throws ArgumentError LuxCore.setup(MersenneTwister(1), layer)
        end
        layer = Parameter(prob,
            constructor(1; default = (rng, times::AbstractVector) -> -times,
                lowerbound = 0.0) => [1.0])
        @test_throws ArgumentError LuxCore.setup(MersenneTwister(1), layer)
    end
    # An exception inside a valid callback must propagate, rather than triggering
    # a second attempt with a different callback argument.
    broken = (rng, n) -> throw(DomainError(n, "initializer failed"))
    layer = InitialConditions(prob, Tunable(1; default = broken) => [1.0])
    @test_throws DomainError LuxCore.setup(MersenneTwister(1), layer)
end

@testset "Shooting solves with initialized reset and control values" begin
    prob = ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 3.0), [1.0])
    layer = ShootingLayer(prob, Tsit5(),
        Tunable(1; kind = Val(:variable), default = (rng, times::AbstractVector) -> 10 .* times) => [1.0, 2.0],
        Constant(1; kind = Val(:parameter), default = [1.0, 2.0, 3.0]) => [1.0, 2.0])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    trajectory, _ = layer(nothing, ps, st)
    @test last(trajectory.u) ≈ [23.0]
    @test reduce(vcat, trajectory.residuals) ≈ [9.0, 8.0]
end
