using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test

@testset "Single-variable initial conditions" begin
    for grid in ([1, 3], [1.0, 3.0])
        layer = Corleone.InitialCondition(grid, (rng, n) -> [10.0, 30.0])
        ps, st = LuxCore.setup(MersenneTwister(1), layer)
        @test length(ps.parameter.ps) == length(grid)
        for (t, expected) in ((0.0, 7.0), (0.5, 7.0), (1.0, 10.0),
                (2.0, 7.0), (3.0, 30.0), (4.0, 7.0))
            value, next_st = layer((7.0, t), ps, st)
            @test value == expected
            @test next_st.lookup == st.lookup
            @test next_st.parameter == st.parameter
            @test next_st.active == (t in grid)
        end
        @test first(layer((7.0, nextfloat(1.0)), ps, st)) == 7.0
        @test isequal(first(layer((-0.0, 0.5), ps, st)), -0.0)
        ps.parameter.ps[2] = 40.0
        @test first(layer((7.0, 3.0), ps, st)) == 40.0
    end
    for grid in (nothing, Float64[])
        layer = Corleone.InitialCondition(grid, 99.0)
        ps, st = LuxCore.setup(MersenneTwister(1), layer)
        @test isempty(ps.parameter.ps)
        @test isempty(Corleone.get_timestops(layer, ps, st, Float64))
        @test first(layer((7.0, 1.0), ps, st)) == 7.0
    end
    @test_throws ArgumentError Corleone.InitialCondition([1.0, 1.0], 0.0)
end

@testset "Independent variable grids and pass-through" begin
    prob = ODEProblem((u, p, t) -> ones(3), [1.0, 2.0, 3.0], (0.0, 4.0), [0.0])
    layer = InitialConditions(prob,
        Tunable(1; default = 10.0) => [1.0, 3.0],
        Constant(2; default = 20.0) => [2.0],
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test keys(layer.initialconditions) == (:u₁, :u₂, :u₃)
    @test length(ps.initialconditions.u₁.parameter.ps) == 2
    @test ps.initialconditions.u₂ == (;)
    @test st.initialconditions.u₂.parameters.parameter.ps == [20.0]
    @test ps.initialconditions.u₃ == (;)
    @test st.initialconditions.u₃ == (;)
    @test Corleone.get_timestops(layer, ps, st, Float64) == [1.0, 2.0, 3.0]
    @test st.active == [false, false, false]
    @test st.active isa Vector{Bool}
    incoming = [4.0, 5.0, 6.0]
    for (t, expected, residuals) in (
            (0.0, incoming, Float64[]),
            (1.0, [10.0, 5.0, 6.0], [6.0]),
            (2.0, [4.0, 20.0, 6.0], Float64[]),
            (3.0, [10.0, 5.0, 6.0], [6.0]),
            (4.0, incoming, Float64[]),
        )
        (value, residual), next_st = layer((incoming, t), ps, st)
        @test value == expected
        @test residual == residuals
        @test next_st.active === st.active
        @test next_st.active == [t in (1.0, 3.0), false, false]
        @test next_st.initialconditions.u₁.active == (t in (1.0, 3.0))
        @test next_st.initialconditions.u₂.states.active == (t == 2.0)
        @test next_st.initialconditions.u₃ == (;)
        @test incoming == [4.0, 5.0, 6.0]
    end
    ps.initialconditions.u₁.parameter.ps .= [11.0, 33.0]
    @test first(first(layer((incoming, 1.0), ps, st))) == [11.0, 5.0, 6.0]
    @test first(first(layer((incoming, 3.0), ps, st))) == [33.0, 5.0, 6.0]
    passthrough = InitialConditions(prob)
    pp, ss = LuxCore.setup(MersenneTwister(1), passthrough)
    @test first(passthrough((incoming, 2.0), pp, ss)) == (incoming, Float64[])
    @test_throws ArgumentError InitialConditions(prob, 1 => [1.0], 1 => [2.0])
    @test_throws ArgumentError InitialConditions(prob, Tunable(1; kind = Val(:parameter)) => [1.0])
end

@testset "Scheduled initial conditions through ShootingLayer" begin
    prob = ODEProblem((u, p, t) -> [p[1], 1.0], [1.0, 2.0], (0.0, 3.0), [1.0])
    layer = ShootingLayer(prob, Tsit5(),
        Tunable(1; kind = Val(:variable), default = 10.0) => [1.0, 2.0],
        Constant(1; kind = Val(:parameter), default = 2.0) => nothing,
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    ps.initialcondition.initialconditions.u₁.parameter.ps .= [10.0, 20.0]
    traj, _ = layer(nothing, ps, st)
    @test traj.t == [0.0, 1.0, 1.0, 2.0, 2.0, 3.0]
    @test traj.reset_indices == [3, 5]
    @test traj.u ≈ [[1.0, 2.0], [3.0, 3.0], [10.0, 3.0],
        [12.0, 4.0], [20.0, 4.0], [22.0, 5.0]]
    @test isempty(traj.residuals[1])
    @test traj.residuals[2] ≈ [7.0]
    @test traj.residuals[3] ≈ [8.0]
    @test traj.ps === ps
end
