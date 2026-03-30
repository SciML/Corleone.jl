using Test
using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using Random
using LuxCore
using SymbolicIndexingInterface

rng = Random.default_rng()

@testset "MTK Example" begin
    @variables begin
        x(t) = 1.0, [tunable = false, bounds = (0.0, 1.0)]
        u(t) = 1.0, [input = true, bounds = (0.0, 1.0)]
    end
    @parameters begin
        p = 1.0, [bounds = (-1.0, 1.0)]
    end
    eqs = [D(x) ~ p * x - u]
    @named simple = ODESystem(eqs, t)

    layer = SingleShootingLayer(simple, [], u => 0.0:0.1:1.0, algorithm = Tsit5(), tspan = (0.0, 1.0))
    ps, st = LuxCore.setup(rng, layer)

    traj, st = layer(nothing, ps, st)

    @testset "MTK symbolic access works" begin
        # This should work with MTK symbols
        u_vals = traj.ps[u]
        @test length(u_vals) == length(traj.t)

        # Controls are observed but not plain parameters
        @test SymbolicIndexingInterface.is_observed(traj, u) == true
        @test SymbolicIndexingInterface.is_parameter(traj, u) == false
    end
end
