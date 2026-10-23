using Corleone
using LuxCore
using OrdinaryDiffEqTsit5
using Random
using Test
import SymbolicIndexingInterface as SII

@testset "Legacy constructor returns ShootingLayer" begin
    prob = ODEProblem((u, p, t) -> [p[1] + p[2]], [1.0], (0.0, 2.0), [2.0, 3.0])
    layer = SingleShootingLayer(prob, Tsit5())
    @test layer isa ShootingLayer
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test keys(ps) == (:initialcondition, :parameter)
    @test isempty(only(ps.initialcondition.initialconditions))
    traj, _ = layer(nothing, ps, st)
    residuals = traj.residuals
    @test first(traj.t) == 0.0
    @test last(traj.t) == 2.0
    @test last(traj.u) ≈ [11.0]
    @test all(isempty, residuals)

    # Input order differs from the underlying parameter order. The second
    # control changes before the first; the grids omit the final endpoint.
    c1 = ControlParameter([0.0, 1.0, 2.0]; controls = [2.0, 4.0, 100.0])
    c2 = ControlParameter([0.0, 0.5]; controls = [3.0, 5.0])
    layer = SingleShootingLayer(prob, Tsit5(); controls = [2 => c2, 1 => c1])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test LuxCore.parameterlength(layer) == 4
    traj, _ = layer(nothing, ps, st)
    @test traj isa Trajectory
    @test SII.parameter_timeseries(traj, 1) == [0.0, 0.5, 1.0]
    @test traj.t == [0.0, 0.5, 1.0, 2.0]
    @test traj.u ≈ [[1.0], [3.5], [7.0], [16.0]]
    @test SII.state_values(only(SII.get_parameter_timeseries_collection(traj))) == [[2.0, 3.0], [2.0, 5.0], [4.0, 5.0]]
    @test last(traj.u) ≈ [16.0]
end

@testset "Initialization and constructor keywords" begin
    prob = ODEProblem((u, p, t) -> p .* u, [1.0, 2.0], (0.0, 1.0), [3.0, 4.0])
    seen = Symbol[]
    init_ic = (rng, prob, indices, bounds) -> begin
        push!(seen, :ic)
        @test indices == [2, 1]
        [8.0, 7.0]
    end
    init_p = (rng, prob, indices, bounds) -> begin
        push!(seen, :p)
        @test indices == [1, 2]
        [5.0, 6.0]
    end
    layer = SingleShootingLayer(
        prob, Tsit5(); tunable_ic = [2, 1],
        state_initialization = init_ic, parameter_initialization = init_p,
        tspan = (0.0, 0.5), quadrature_indices = [2]
    )
    @test isempty(seen)
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test seen == [:ic, :p]
    @test collect(values(only(ps.initialcondition.initialconditions))) == [7.0, 8.0]
    traj, _ = layer(nothing, ps, st)
    residuals = traj.residuals
    @test only(residuals) == [6.0, 6.0]
    @test first(traj.u) == [7.0, 8.0]
    @test last(traj.t) == 0.5
    @test last(traj.u) ≈ [7exp(2.5), 8exp(3.0)] rtol = 1.0e-3

    layer = SingleShootingLayer(
        prob, Tsit5(); tunable_ic = [2],
        bounds_ic = ([0.0, 5.0], [10.0, 6.0]), bounds_p = ([0.0, 0.0], [1.0, 2.0])
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test only(values(only(ps.initialcondition.initialconditions))) == 5.0
    traj, _ = layer(nothing, ps, st)
    residuals = traj.residuals
    @test only(residuals) == [3.0]
    @test first(traj.u) == [1.0, 5.0]
    @test SII.parameter_values(traj) == [1.0, 2.0]

    @test_throws ArgumentError SingleShootingLayer(prob, Tsit5(); tunable_ic = [3])
    @test_throws ArgumentError SingleShootingLayer(prob, Tsit5(); quadrature_indices = [3])
    c = ControlParameter([0.0]; controls = [1.0])
    @test_throws ArgumentError SingleShootingLayer(prob, Tsit5(); controls = [1 => c, 1 => c])
    @test_throws ArgumentError SingleShootingLayer(prob, Tsit5(); bounds_p = ([0.0], [1.0]))
end

@testset "Control initializers and empty parameters" begin
    prob = ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 1.0), [0.0])
    calls = Ref(0)
    c = ControlParameter(
        [0.25, 0.5, 1.0]; bounds = (0.0, 1.0),
        controls = (rng, t, bounds) -> begin
            calls[] += 1
            @test t == [0.25, 0.5]
            @test bounds == ([0.0, 0.0], [1.0, 1.0])
            rand(rng, length(t))
        end
    )
    layer = SingleShootingLayer(prob, Tsit5(); controls = (1 => c,))
    rng = MersenneTwister(42)
    expected = rand(copy(rng), 2)
    ps, st = LuxCore.setup(rng, layer)
    @test calls[] == 1
    @test ps.parameter.parameters.p₁.parameter.ps == expected
    traj, _ = layer(nothing, ps, st)
    @test last(traj.u) ≈ [sum(expected) / 2]

    prob = ODEProblem((u, p, t) -> -u, [1.0], (0.0, 1.0), Float64[])
    layer = SingleShootingLayer(prob, Tsit5())
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    traj, _ = layer(nothing, ps, st)
    @test last(traj.u) ≈ [exp(-1)] rtol = 1.0e-3
    @test isempty(ps.parameter.parameters)
end


@testset "Shooting constraints use tunable state indices" begin
    prob = ODEProblem((u, p, t) -> [1.0, 2.0, 3.0], [10.0, 20.0, 30.0], (0.0, 1.0), Float64[])
    ic = Corleone.InitialCondition(prob; fixed = [1, 3])
    ps, st = LuxCore.setup(MersenneTwister(1), ic)
    (u0, residuals), next_st = ic([100.0, 7.0, 300.0], ps, st)
    @test u0 == [10.0, 20.0, 30.0]
    @test residuals == [13.0]
    @test next_st == st

    # At an interior shooting point, compare only the tunable state with the
    # previous segment endpoint, using the physical state order after repacking.
    layer = ShootingLayer(
        prob, Tsit5(),
        0.0 => (; fixed = [1, 2, 3]),
        0.5 => (; variables = (2 => 40.0,), fixed = [1, 3]),
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    traj, _ = layer(nothing, ps, st)
    residuals = traj.residuals
    @test length(traj.residuals) == 2
    @test traj.t == [0.0, 0.5, 0.5, 1.0]
    @test traj.reset_indices == [3]
    @test isempty(first(residuals))
    @test last(residuals) ≈ [19.0]
    @test traj.u[2] ≈ [10.5, 21.0, 31.5]
    @test traj.u[3] == [10.0, 40.0, 30.0]
    @test last(traj.u) ≈ [10.5, 41.0, 31.5]
end
