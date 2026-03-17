using Corleone
using LuxCore
using OrdinaryDiffEqTsit5
using Random
using SciMLBase
using SymbolicIndexingInterface
using Test

rng = MersenneTwister(29)

function lqr2d!(du, u, p, t)
    a, b, uctrl = p
    du[1] = a * u[1] + b * uctrl
    du[2] = u[1]^2 + 0.1 * uctrl^2
    return nothing
end

@testset "MultipleShootingLayer" begin
    prob = ODEProblem(
        ODEFunction(lqr2d!; sys = SymbolCache([:x, :q], [:a, :b, :u], :t)),
        [1.0, 0.0],
        (0.0, 6.0),
        [-0.3, 1.0, 0.0],
    )

    controls = (
        Corleone.FixedControlParameter([0.0]; name = :a, controls = (rng, t) -> [-0.3]),
        Corleone.FixedControlParameter([0.0]; name = :b, controls = (rng, t) -> [1.0]),
        ControlParameter(
            collect(0.0:0.5:5.5);
            name = :u,
            controls = (rng, t) -> fill(0.2, length(t)),
            bounds = t -> (zero(t) .- 2.0, zero(t) .+ 2.0),
        ),
    )

    single = SingleShootingLayer(prob, controls...; algorithm = Tsit5(), name = :single_lqr, quadrature_indices = [2])

    @testset "Construction and Shooting Variables" begin
        mlayer = MultipleShootingLayer(single, 2.0, 4.0; ensemble_algorithm = SciMLBase.EnsembleSerial())

        @test keys(mlayer.layer.layers) == (:layer_1, :layer_2, :layer_3)
        @test Corleone.get_quadrature_indices(mlayer) == [2]

        sv = mlayer.shooting_variables
        @test sv.layer_1.state == Int[]
        @test sv.layer_2.state == [1]
        @test sv.layer_3.state == [1]
        @test sv.layer_1.control == []
        @test sv.layer_2.control == []
        @test sv.layer_3.control == []
    end

    @testset "Evaluation and Matching Constraints" begin
        mlayer = MultipleShootingLayer(single, 2.0, 4.0; ensemble_algorithm = SciMLBase.EnsembleSerial())
        ps, st = LuxCore.setup(rng, mlayer)

        traj, st2 = mlayer(nothing, ps, st)
        @test traj isa Corleone.Trajectory
        @test st2 isa NamedTuple
        @test keys(traj.shooting) == (:matching_1, :matching_2)

        # Build expected matching values directly from the underlying parallel solutions.
        parts = @inferred first(mlayer.layer(nothing, ps, st))
        expected_state_1 = first(parts.layer_2.u)[1] - last(parts.layer_1.u)[1]
        expected_state_2 = first(parts.layer_3.u)[1] - last(parts.layer_2.u)[1]

        @test isapprox(traj.shooting.matching_1.state.x, expected_state_1; atol = 1.0e-12)
        @test isapprox(traj.shooting.matching_2.state.x, expected_state_2; atol = 1.0e-12)

        @test first(traj.t) == 0.0
        @test isapprox(last(traj.t), 6.0; atol = 1.0e-12)
        @test length(traj.u) == length(traj.t)
    end

    @testset "Remake" begin
        mlayer = MultipleShootingLayer(single, 2.0, 4.0; ensemble_algorithm = SciMLBase.EnsembleSerial())
        remade = remake(mlayer; layer_2 = (; name = :middle_changed))

        @test remade.layer.layers.layer_2.name == :middle_changed
    end
end
