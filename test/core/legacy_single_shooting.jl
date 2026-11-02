using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test
import SymbolicIndexingInterface as SII

@testset "Legacy constructors are functions returning native objects" begin
    @test ControlParameter isa Function
    @test SingleShootingLayer isa Function
    @test MultipleShootingLayer isa Function
    @test !isdefined(Corleone, :LegacyInitializer)
    definition = @test_deprecated r"ControlParameter" ControlParameter(
        [0.0, 1.0];
        name = :p₁, controls = [2.0, 4.0], bounds = (0.0, 5.0)
    )
    @test definition isa Pair
    @test first(definition) isa Corleone.AbstractAnnotation
    @test last(definition) == [1.0]
    prob = ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 2.0), [1.0])
    direct = ShootingLayer(prob, Tsit5(), definition)
    ps, st = LuxCore.setup(MersenneTwister(1), direct)
    @test ps.parameter.parameters.p₁.parameter.ps == [2.0, 4.0]
    @test last(first(direct(nothing, ps, st)).u) ≈ [6.0]
    layer = @test_deprecated r"SingleShootingLayer" SingleShootingLayer(prob, Tsit5())
    @test layer isa ShootingLayer
    layer = @test_deprecated r"MultipleShootingLayer" MultipleShootingLayer(prob, Tsit5(), 1.0)
    @test layer isa ShootingLayer
end

@testset "Legacy single-shooting translation" begin
    prob = ODEProblem((u, p, t) -> [p[1] + p[2]], [1.0], (0.0, 2.0), [2.0, 3.0])
    layer = SingleShootingLayer(prob, Tsit5())
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test ps.initialcondition.initialconditions.u₁ == (;)
    traj, _ = layer(nothing, ps, st)
    @test last(traj.u) ≈ [11.0]
    @test all(isempty, traj.residuals)

    c1 = ControlParameter([0.0, 1.0, 2.0]; controls = [2.0, 4.0, 100.0])
    c2 = ControlParameter([0.0, 0.5]; controls = [3.0, 5.0])
    layer = SingleShootingLayer(prob, Tsit5(); controls = [2 => c2, 1 => c1])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test LuxCore.parameterlength(layer) == 4
    @test ps.parameter.parameters.p₁.parameter.ps == [2.0, 4.0]
    @test ps.parameter.parameters.p₂.parameter.ps == [3.0, 5.0]
    traj, _ = layer(nothing, ps, st)
    @test traj.t == [0.0, 0.5, 1.0, 2.0]
    @test traj.u ≈ [[1.0], [3.5], [7.0], [16.0]]
    @test control_indices(layer.sys) == [2, 1]
end

@testset "Bounds, tunable starts, and remade problems" begin
    prob = ODEProblem((u, p, t) -> p .* u, [1.0, 2.0], (0.0, 1.0), [3.0, 4.0])
    layer = SingleShootingLayer(
        prob, Tsit5(); tunable_ic = [2],
        bounds_ic = ([0.0, 5.0], [10.0, 6.0]), bounds_p = ([0.0, 0.0], [1.0, 2.0]),
        quadrature_indices = [2], tspan = (0.0, 0.5), saveat = [0.25]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test ps.initialcondition.initialconditions.u₂.parameter.ps == [5.0]
    @test ps.initialcondition.initialconditions.u₁ == (;)
    @test ps.parameter.parameters.p₁.parameter.p₁.ps == [1.0]
    @test ps.parameter.parameters.p₁.parameter.p₂.ps == [2.0]
    traj, _ = layer(nothing, ps, st)
    @test first(traj.u) == [1.0, 5.0]
    @test isempty(only(traj.residuals))
    @test 0.25 in traj.t
    @test last(traj.t) == 0.5
    @test last(traj.u) ≈ [exp(0.5), 5exp(1.0)] rtol = 1.0e-3
    @test quadrature_indices(layer.sys) == [2]
    @test Corleone.get_bounds(layer.sys, :u₂) == (5.0, 6.0)
    @test Corleone.get_bounds(layer.sys, :p₂) == (0.0, 2.0)
end

@testset "Legacy control callbacks and repeated setup" begin
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
    layer = SingleShootingLayer(prob, Tsit5(); controls = [1 => c])
    @test calls[] == 0
    @test LuxCore.parameterlength(layer) == 2
    @test calls[] == 0
    ps, st = LuxCore.setup(MersenneTwister(42), layer)
    @test calls[] == 1
    expected = rand(MersenneTwister(42), 2)
    @test ps.parameter.parameters.p₁.parameter.ps == expected
    again, _ = LuxCore.setup(MersenneTwister(42), layer)
    @test calls[] == 2
    @test again == ps
    traj, _ = layer(nothing, ps, st)
    @test last(traj.u) ≈ [sum(expected) / 2]
    @test calls[] == 2
end

@testset "Legacy multiple shooting uses native reset schedules" begin
    prob = ODEProblem((u, p, t) -> [p[1], u[1]], [1.0, 0.0], (0.0, 3.0), [2.0])
    control = ControlParameter([0.0, 1.0, 2.0]; controls = [2.0, 3.0, 4.0])
    base = SingleShootingLayer(prob, Tsit5(); controls = [1 => control], tunable_ic = [1], quadrature_indices = [2])
    layer = MultipleShootingLayer(base, [0.0, 1.0, 2.0, 3.0]; ensemble_alg = EnsembleThreads())
    @test layer isa ShootingLayer
    @test layer.parameter === base.parameter
    @test layer.sys === base.sys
    @test layer.ensemblealg isa EnsembleThreads
    @test quadrature_indices(layer.sys) == [2]
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    @test ps.initialcondition.initialconditions.u₁.parameter.ps == [1.0, 1.0, 1.0]
    @test ps.initialcondition.initialconditions.u₂.parameter.ps == [0.0, 0.0]
    serial = ShootingLayer(layer.name, prob, layer.alg, EnsembleSerial(), layer.initialcondition, layer.parameter, layer.sys)
    a, _ = layer(nothing, ps, st)
    b, _ = serial(nothing, ps, st)
    @test a.u ≈ b.u
    @test a.t == b.t
    @test a.residuals ≈ b.residuals
    @test a.reset_indices == [3, 5]
    @test length.(a.residuals) == [0, 2, 2]
    @test last(a.u)[1] ≈ 5.0
    other = MultipleShootingLayer(prob, Tsit5(), 1.0, 2.0; controls = [1 => control])
    @test other isa ShootingLayer
    @test Corleone.get_timestops(
        other.initialcondition,
        LuxCore.initialparameters(MersenneTwister(1), other).initialcondition,
        LuxCore.initialstates(MersenneTwister(1), other).initialcondition, Float64
    ) == [1.0, 2.0]
    empty = SingleShootingLayer(ODEProblem((u, p, t) -> -u, [1.0], (0.0, 1.0), Float64[]), Tsit5())
    p0, s0 = LuxCore.setup(MersenneTwister(1), empty)
    @test last(first(empty(nothing, p0, s0)).u) ≈ [exp(-1)] rtol = 1.0e-3
end

@testset "Legacy translation rejects unsupported or invalid definitions" begin
    prob = ODEProblem((u, p, t) -> p, [1.0], (0.0, 2.0), [1.0])
    @test_throws ArgumentError ControlParameter(Float64[])
    @test_throws ArgumentError ControlParameter([1.0, 0.0])
    @test_throws ArgumentError ControlParameter([0.0, 0.0])
    @test_throws ArgumentError ControlParameter([0.0, 1.0]; bounds = ([0.0, 0.5], [1.0, 1.0]))
    @test_throws DimensionMismatch ControlParameter([0.0, 1.0]; controls = [0.0])
    @test_throws ArgumentError SingleShootingLayer(prob, Tsit5(); tunable_ic = [2])
    @test_throws ArgumentError SingleShootingLayer(prob, Tsit5(); tunable_ic = [1, 1])
    @test_throws ArgumentError SingleShootingLayer(prob, Tsit5(); quadrature_indices = [2])
    @test_throws ArgumentError SingleShootingLayer(prob, Tsit5(); controls = [2 => ControlParameter([0.0])])
    c = ControlParameter([0.0])
    @test_throws ArgumentError SingleShootingLayer(prob, Tsit5(); controls = [1 => c, 1 => c])
    @test_throws ArgumentError SingleShootingLayer(prob, Tsit5(); bounds_p = ([], []))
    @test_throws ArgumentError SingleShootingLayer(prob, Tsit5(); state_initialization = identity)
    @test_throws ArgumentError SingleShootingLayer(prob, Tsit5(); parameter_initialization = identity)
    @test_throws ArgumentError MultipleShootingLayer(prob, Tsit5(), 1.0; initialization = identity)
    @test_throws ArgumentError MultipleShootingLayer(prob, Tsit5(), -1.0)
    fixed = ShootingLayer(prob, Tsit5(), Constant(1; kind = Val(:variable)) => [0.0])
    @test_throws ArgumentError MultipleShootingLayer(fixed, 1.0)
end
