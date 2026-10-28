using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test

@testset "Shooting guesses from a reference solve" begin
    prob = ODEProblem((u, p, t) -> p .* u, [1.0, 2.0], (0.0, 3.0), [0.2, -0.1]; abstol = 1.0e-12, reltol = 1.0e-12)
    times = [0.5, 1.5, 2.0]
    reference = solve(prob, Tsit5())
    definitions = map(1:2) do i
        Tunable(i; kind = Val(:variable), default = (rng, times::AbstractVector) -> [reference(t)[i] for t in times]) => times
    end
    for alg in (EnsembleSerial(), EnsembleThreads())
        layer = ShootingLayer(prob, Tsit5(), definitions...; ensemblealg = alg)
        ps, st = LuxCore.setup(MersenneTwister(42), layer)
        for (i, entry) in enumerate(values(ps.initialcondition.initialconditions.u₁.parameter))
            @test entry.ps ≈ [prob.u0[i] * exp(prob.p[i] * t) for t in times] rtol = 1.0e-9
        end
        trajectory, _ = layer(nothing, ps, st)
        @test first(trajectory.u) == prob.u0
        @test length(shooting_constraints(trajectory)) == 6
        @test maximum(abs, shooting_constraints(trajectory)) < 1.0e-9
        @test last(trajectory.u) ≈ reference(3.0) rtol = 1.0e-9
    end
end

@testset "Mixed reference and linear guesses on separate grids" begin
    prob = ODEProblem((u, p, t) -> p .* u, [1.0, 2.0], (2.0, 6.0), [0.2, -0.1])
    reference = solve(prob, Tsit5(); abstol = 1.0e-12, reltol = 1.0e-12)
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable), default = (rng, times::AbstractVector) -> [reference(t)[1] for t in times]) => [3.0, 5.0],
        Tunable(2; kind = Val(:variable), default = (rng, times::AbstractVector) -> 2 .+ (4 .- 2) .* (times .- 2) ./ (6 - 2)) => [4.0, 5.0]
    )
    ps, st = LuxCore.setup(MersenneTwister(42), layer)
    @test ps.initialcondition.initialconditions.u₁.parameter.ps ≈ exp.(0.2 .* [1.0, 3.0]) rtol = 1.0e-9
    @test ps.initialcondition.initialconditions.u₂.parameter.ps == [3.0, 3.5]
    @test get_number_of_shooting_constraints(layer, ps, st) == 4
end

@testset "Explicit shooting initialization helpers" begin
    rng = MersenneTwister(42)
    prob = ODEProblem((u, p, t) -> p .* u, [1.0, 2.0, 3.0], (2.0, 6.0), [0.2, -0.1, 0.0]; abstol = 1.0e-11, reltol = 1.0e-11)
    for alg in (EnsembleSerial(), EnsembleThreads())
        layer = ShootingLayer(
            prob, Tsit5(),
            Tunable(1; kind = Val(:variable), lowerbound = 0.0, upperbound = 10.0) => [2.0, 3.0, 5.0, 6.0],
            Tunable(2; kind = Val(:variable), lowerbound = 0.0, upperbound = 10.0) => [4.0, 5.0],
            Constant(3; kind = Val(:variable), default = 4.0) => [4.0],
            Tunable(1; kind = Val(:parameter), default = [0.2, 0.1]) => [4.5]; ensemblealg = alg
        )
        ps, st = LuxCore.setup(rng, layer)
        x = ps.initialcondition.initialconditions.u₁.parameter.ps
        y = ps.initialcondition.initialconditions.u₂.parameter.ps
        original_st = deepcopy(st)
        original_parameters = deepcopy(ps.parameter)
        @test constant_initialization(rng, layer, ps, st; u0 = [2.0, 3.0, 9.0]) === ps
        @test x == fill(2.0, 4)
        @test y == fill(3.0, 2)
        @test linear_initialization(rng, layer, ps, st; u_infinity = [5.0, 6.0, 9.0]) === ps
        @test x == [1.0, 2.0, 4.0, 5.0]
        @test y == [4.0, 5.0]
        @test custom_initialization(rng, layer, ps, st; u0s = Dict(2 => [7.0, 8.0])) === ps
        @test y == [7.0, 8.0]
        @test x == [1.0, 2.0, 4.0, 5.0]
        before = copy(x)
        @test_throws DimensionMismatch custom_initialization(rng, layer, ps, st; u0s = Dict(1 => [1.0]))
        @test_throws ArgumentError constant_initialization(rng, layer, ps, st; u0 = [11.0, 1.0, 1.0])
        @test x == before
        random_initialization(MersenneTwister(7), layer, ps, st)
        random_x, random_y = copy(x), copy(y)
        random_initialization(MersenneTwister(7), layer, ps, st)
        @test x == random_x && y == random_y
        @test all(v -> 0 <= v <= 10, vcat(x, y))
        @test hybrid_initialization(
            rng, layer, ps, st,
            1 => ((rng, l, p, s; variables) -> constant_initialization(rng, l, p, s; variables, u0 = [2.0])),
            2 => ((rng, l, p, s; variables) -> linear_initialization(rng, l, p, s; variables, u_infinity = [0.0, 6.0]))
        ) === ps
        @test x == fill(2.0, 4)
        @test y == [4.0, 5.0]
        @test forward_initialization(rng, layer, ps, st) === ps
        @test x ≈ 2 .* exp.(0.2 .* [0.0, 1.0, 2.5, 2.5] .+ 0.1 .* [0.0, 0.0, 0.5, 1.5]) rtol = 1.0e-9
        @test y ≈ 2 .* exp.(-0.1 .* [2.0, 3.0]) rtol = 1.0e-9
        @test ps.initialcondition.initialconditions.u₁.parameter.ps === x
        @test ps.parameter == original_parameters
        @test st == original_st
        trajectory, _ = layer(nothing, ps, st)
        @test maximum(abs, shooting_constraints(trajectory)) < 1.0e-8
        @test last(trajectory.u)[3] == 4.0
    end
end

@testset "Initialization uses current state schedules and structured parameters" begin
    using ComponentArrays
    prob = ODEProblem((u, p, t) -> p .* u, [1.0], (0.0, 3.0), [0.2]; abstol = 1.0e-12, reltol = 1.0e-12)
    layer = ShootingLayer(prob, Tsit5(), Tunable(1; kind = Val(:variable)) => [1.0, 2.0])
    rng = MersenneTwister(42)
    ps, st = LuxCore.setup(rng, layer)
    ps = ComponentArray(ps)
    collection = st.initialcondition.initialconditions.u₁.lookup.collection
    delete!(collection, 1.0)
    collection[0.5] = 1
    before = deepcopy(st)
    forward_initialization(rng, layer, ps, st)
    @test ps.initialcondition.initialconditions.u₁.parameter.ps ≈ exp.(0.2 .* [0.5, 2.0]) rtol = 1.0e-9
    @test st == before
    old = copy(ps)
    @test_throws ArgumentError random_initialization(rng, layer, ps, st)
    @test ps == old
    @test_throws ArgumentError constant_initialization(rng, layer, ps, st; variables = [2])
    plain = ShootingLayer(prob, Tsit5())
    plain_ps, plain_st = LuxCore.setup(rng, plain)
    @test forward_initialization(rng, plain, plain_ps, plain_st) === plain_ps
end

@testset "Forward initialization retains fixed and unselected resets" begin
    rng = MersenneTwister(12)
    prob = ODEProblem((u, p, t) -> [u[2], 0.0], [0.0, 1.0], (0.0, 3.0), Float64[])
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable), default = 0.0) => [1.0, 2.0, 3.0],
        Constant(2; kind = Val(:variable), default = 2.0) => [1.5]
    )
    ps, st = LuxCore.setup(rng, layer)
    forward_initialization(rng, layer, ps, st)
    @test ps.initialcondition.initialconditions.u₁.parameter.ps ≈ [1.0, 2.5, 4.5]
    other = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable), default = 0.0) => [1.0, 2.0, 3.0],
        Tunable(2; kind = Val(:variable), default = 3.0) => [1.5]
    )
    ps, st = LuxCore.setup(rng, other)
    forward_initialization(rng, other, ps, st; variables = [1])
    @test ps.initialcondition.initialconditions.u₁.parameter.ps ≈ [1.0, 3.0, 6.0]
    @test ps.initialcondition.initialconditions.u₂.parameter.ps == [3.0]
    bounded = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable), default = 0.0, lowerbound = 0.0, upperbound = 0.5) => [1.0, 2.0]
    )
    ps, st = LuxCore.setup(rng, bounded)
    before_ps, before_st = deepcopy(ps), deepcopy(st)
    @test_throws ArgumentError forward_initialization(rng, bounded, ps, st)
    @test ps == before_ps
    @test st == before_st
end
