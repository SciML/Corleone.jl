using Corleone
using LuxCore
using OrdinaryDiffEqTsit5
using Random
using SciMLBase
using SymbolicIndexingInterface
using Test

rng = MersenneTwister(21)

function lqr2d!(du, u, p, t)
    a, b, uctrl = p
    du[1] = a * u[1] + b * uctrl
    du[2] = u[1]^2 + 0.1 * uctrl^2
    return nothing
end

function make_layer(uctrl; name)
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
            controls = (rng, t) -> fill(uctrl, length(t)),
            bounds = t -> (zero(t) .- 2.0, zero(t) .+ 2.0),
        ),
    )

    SingleShootingLayer(prob, controls...; algorithm = Tsit5(), name = name, quadrature_indices = [2])
end

@testset "ParallelShootingLayer" begin
    layer1 = make_layer(0.1; name = :ss1)
    layer2 = make_layer(0.4; name = :ss2)

    @testset "Construction and Block Structure" begin
        parallel = ParallelShootingLayer(layer1, layer2; ensemble_algorithm = SciMLBase.EnsembleSerial())
        @test parallel.layers isa NamedTuple
        @test keys(parallel.layers) == (:layer1, :layer2)

        p1 = LuxCore.parameterlength(layer1)
        p2 = LuxCore.parameterlength(layer2)
        @test Corleone.get_block_structure(parallel) == [0, p1, p1 + p2]
    end

    @testset "Evaluation and Output Shapes" begin
        parallel = ParallelShootingLayer(layer1, layer2; ensemble_algorithm = SciMLBase.EnsembleSerial())
        ps, st = LuxCore.setup(rng, parallel)

        out, st2 = parallel(nothing, ps, st)
        @test out isa NamedTuple
        @test st2 isa NamedTuple
        @test keys(out) == (:layer1, :layer2)
        @test keys(st2) == (:layer1, :layer2)
        @test out.layer1 isa Corleone.Trajectory
        @test out.layer2 isa Corleone.Trajectory
        @test first(out.layer1.t) == 0.0
        @test first(out.layer2.t) == 0.0
        @test isapprox(last(out.layer1.t), 6.0; atol = 1.0e-12)
        @test isapprox(last(out.layer2.t), 6.0; atol = 1.0e-12)

        # Different control values should produce different trajectories.
        @test out.layer1.u[end][1] != out.layer2.u[end][1]
    end

    @testset "Remake" begin
        parallel = ParallelShootingLayer(layer1, layer2; ensemble_algorithm = SciMLBase.EnsembleSerial())
        remade = remake(parallel; layer1 = (; name = :changed_layer), ensemble_algorithm = SciMLBase.EnsembleSerial())

        @test remade.layers.layer1.name == :changed_layer
        @test remade.layers.layer2.name == parallel.layers.layer2.name
    end
end
