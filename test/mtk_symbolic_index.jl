# Test for MTK symbolic indexing fix
# This test validates that traj.ps[mtk_symbol] works correctly with MTK symbolic variables

using Test
using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using Random
using LuxCore
using SymbolicIndexingInterface

rng = Random.default_rng()

@testset "MTK Symbolic Indexing" begin
    # Define MTK system with input control
    @variables begin
        x(t) = 1.0, [tunable = false, bounds = (0.0, 2.0)]
        u(t) = 1.0, [input = true, bounds = (0.0, 2.0)]
    end
    @parameters begin
        p = 1.0, [bounds = (-1.0, 1.0)]
    end

    eqs = [D(x) ~ p * x - u]
    @named simple = ODESystem(eqs, t)

    # Create SingleShootingLayer with MTK system
    # Note: Both p and u become ControlParameters by design
    layer = SingleShootingLayer(
        simple,
        [],
        u => 0.0:0.1:1.0;
        algorithm = Tsit5(),
        tspan = (0.0, 1.0)
    )

    ps, st = LuxCore.setup(rng, layer)
    traj, st2 = layer(nothing, ps, st)

    @testset "Control names are Symbols (MTK symbols converted)" begin
        # Both p and u(t) are controls
        # _maybesymbolifyme extracts base symbol :u from u(t)
        control_names = Corleone._control_names(traj)
        @test all(name -> name isa Symbol, control_names)
        # Check that :u (base symbol extracted from u(t)) is in control names
        @test :u in control_names
        @test :p in control_names
    end

    @testset "is_observed with MTK symbols" begin
        # Both u and p are observed (they are ControlParameters)
        @test SymbolicIndexingInterface.is_observed(traj, u) == true
        @test SymbolicIndexingInterface.is_observed(traj, p) == true
        # x is a state, not a control
        @test SymbolicIndexingInterface.is_observed(traj, x) == false
    end

    @testset "is_parameter distinguishes controls from states" begin
        # Both u and p are observed (controls), so not plain parameters
        @test SymbolicIndexingInterface.is_parameter(traj, u) == false
        @test SymbolicIndexingInterface.is_parameter(traj, p) == false
        # x is a state, not a parameter
        @test SymbolicIndexingInterface.is_parameter(traj, x) == false
    end

    @testset "traj.ps[mtk_symbol] returns values" begin
        # Using MTK symbols - should work now
        u_vals = traj.ps[u]
        @test length(u_vals) == length(traj.t)

        # p is also a ControlParameter, returns values over time
        p_vals = traj.ps[p]
        @test length(p_vals) == length(traj.t)
    end

    @testset "getsym works for state and control" begin
        # State via getsym
        x_vals = getsym(traj, x)(traj)
        @test length(x_vals) == length(traj.t)

        # Control via getsym
        u_vals = getsym(traj, u)(traj)
        @test length(u_vals) == length(traj.t)
    end
end
