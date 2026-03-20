using Corleone
using CorleoneOED
using OrdinaryDiffEqTsit5
using SciMLBase
using SymbolicIndexingInterface
using Test
using Random
using LuxCore

@testset "Augmentation V2 - New API" begin
    # Build a simple decay problem
    function f(du, u, p, t)
        du[1] = -p[1] * u[1]
    end
    u0 = [1.0]
    tspan = (0.0, 2.0)
    p = [0.5]
    
    # Test with explicit SymbolCache
    prob = ODEProblem{true}(
        ODEFunction(f, sys=SymbolCache([:x], [:k], :t)), u0, tspan, p)
    control = Corleone.ControlParameter(0.0:0.5:2.0, name=:k)
    layer = Corleone.SingleShootingLayer(prob, control; algorithm=Tsit5())
    
    @testset "Get symbolic equations" begin
        sys = get_symbolic_equations(layer)
        @test length(sys.vars) == 1
        @test length(sys.parameters) == 1
        @test Symbol(sys.parameters[1]) == :k
    end
    
    @testset "Append sensitivities" begin
        sys = get_symbolic_equations(layer)
        append_sensitivity!(sys, [:k])
        @test !isnothing(sys.sensitivities)
        @test size(sys.sensitivities) == (1, 1)
        @test length(sys.sensitivity_equations) == 1
    end
    
    @testset "Add measurements" begin
        sys = get_symbolic_equations(layer)
        append_sensitivity!(sys, [:k])
        
        disc_cp = ControlParameter(:w1 => collect(0.0:0.5:2.0))
        cont_cp = ControlParameter(:w2 => collect(0.0:0.1:2.0))
        disc_meas = DiscreteMeasurement(disc_cp, (u, p, t) -> u[1])
        cont_meas = ContinuousMeasurement(cont_cp, (u, p, t) -> u[1])
        
        add_observed!(sys, disc_meas, cont_meas)
        
        @test length(sys.discrete_measurements) == 1
        @test length(sys.continuous_measurements) == 1
        @test !isnothing(sys.fisher_continuous_vars)
        @test length(sys.fisher_continuous_vars) == 1  # 1x1 matrix -> 1 element
    end
    
    @testset "Create augmented layer" begin
        sys = get_symbolic_equations(layer)
        append_sensitivity!(sys, [:k])
        
        disc_cp = ControlParameter(:w1 => collect(0.0:0.5:2.0))
        cont_cp = ControlParameter(:w2 => collect(0.0:0.1:2.0))
        disc_meas = DiscreteMeasurement(disc_cp, (u, p, t) -> u[1])
        cont_meas = ContinuousMeasurement(cont_cp, (u, p, t) -> u[1])
        add_observed!(sys, disc_meas, cont_meas)
        
        new_layer = SingleShootingLayer(sys, layer)
        new_prob = Corleone.get_problem(new_layer)
        
        # Original: 1 state
        # + Sensitivities: 1 state (1x1)
        # + Fisher: 1 state (1x1 symmetric = 1 element)
        # Total: 3 states
        @test length(new_prob.u0) == 3
    end
    
    @testset "Solve and extract Fisher" begin
        sys = get_symbolic_equations(layer)
        append_sensitivity!(sys, [:k])
        
        cont_cp = ControlParameter(:w2 => collect(0.0:0.1:2.0))
        cont_meas = ContinuousMeasurement(cont_cp, (u, p, t) -> u[1])
        add_observed!(sys, cont_meas)
        
        new_layer = SingleShootingLayer(sys, layer)
        oed_layer = OEDLayerV2(sys, new_layer)
        
        rng = Random.default_rng()
        ps = LuxCore.initialparameters(rng, oed_layer)
        st = LuxCore.initialstates(rng, oed_layer)
        (fisher, traj), st_new = oed_layer(nothing, ps, st)
        
        @test length(traj.t) > 0
        @test length(traj.u) == length(traj.t)
        @test fisher isa Matrix
        @test size(fisher) == (1, 1)
        
        # Can also extract Fisher separately
        F = fisher_information(oed_layer, traj)
        @test size(F) == (1, 1)
        @test F[1, 1] > 0  # Should be positive
    end
    
    @testset "Extract sensitivities" begin
        sys = get_symbolic_equations(layer)
        append_sensitivity!(sys, [:k])
        
        cont_cp = ControlParameter(:w2 => collect(0.0:0.1:2.0))
        cont_meas = ContinuousMeasurement(cont_cp, (u, p, t) -> u[1])
        add_observed!(sys, cont_meas)
        
        new_layer = SingleShootingLayer(sys, layer)
        oed_layer = OEDLayerV2(sys, new_layer)
        
        rng = Random.default_rng()
        ps = LuxCore.initialparameters(rng, oed_layer)
        st = LuxCore.initialstates(rng, oed_layer)
        (fisher, traj), st_new = oed_layer(nothing, ps, st)
        
        sens = sensitivities(oed_layer, traj)
        @test length(sens) == length(traj.t)
        @test size(sens[1]) == (1, 1)  # 1 state x 1 parameter
        @test sens[1][1, 1] == 0.0  # Initial sensitivity is 0
    end
end
