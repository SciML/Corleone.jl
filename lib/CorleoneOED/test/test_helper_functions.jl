using Test
using CorleoneOED
using Corleone
using OrdinaryDiffEqTsit5
using Random
using LuxCore
using SymbolicIndexingInterface

@testset "Helper Functions" begin
    # Set up a simple ODE
    function odefn!(du, u, p, t)
        du[1] = -p[1] * u[1]
    end
    
    u0 = [1.0]
    tspan = (0.0, 1.0)
    
    prob = ODEProblem{true}(
        ODEFunction(odefn!, sys=SymbolCache([:x], [:k], :t)),
        u0, tspan, [0.5]
    )
    
    k = ControlParameter(0.0:0.5:1.0, name=:k)
    base_layer = SingleShootingLayer(prob, k; algorithm=Tsit5())
    
    @testset "create_oed_layer" begin
        # Test create_oed_layer
        w = ControlParameter(0.0:0.5:1.0, name=:w)
        oed_layer = create_oed_layer(
            base_layer,
            [:k],
            ContinuousMeasurement(w, (u, p, t) -> u[1])
        )
        
        @test oed_layer isa OEDLayerV2
        @test oed_layer.layer isa SingleShootingLayer
        @test oed_layer.symbolic_system isa SymbolicSystem
        
        # Test that it can be solved
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, oed_layer)
        (F, traj), st = oed_layer(nothing, ps, st)
        
        @test traj isa Trajectory
        @test length(traj.u) > 1
        
        # Test Fisher information extraction
        F_cont = fisher_information(oed_layer, traj)
        @test F_cont isa Matrix
        @test size(F_cont) == (1, 1)
        @test F_cont[1,1] >= 0  # Fisher info should be non-negative
    end
    
    @testset "augment_fisher" begin
        sys = get_symbolic_equations(base_layer)
        append_sensitivity!(sys, [:k])
        
        w = ControlParameter(0.0:0.5:1.0, name=:w)
        augment_fisher(sys, ContinuousMeasurement(w, (u, p, t) -> u[1]))
        
        @test !isnothing(sys.fisher_continuous_vars)
        @test !isnothing(sys.fisher_continuous_eqs)
        
        # Create layer and test
        aug_layer = SingleShootingLayer(sys, base_layer)
        oed_layer = OEDLayerV2(sys, aug_layer)
        
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, oed_layer)
        (F, traj), st = oed_layer(nothing, ps, st)
        
        F_cont = fisher_information(oed_layer, traj)
        @test F_cont isa Matrix
        @test size(F_cont) == (1, 1)
    end
    
    @testset "augment_sensitivities" begin
        sens_layer = augment_sensitivities(base_layer, [:k])
        @test sens_layer isa SingleShootingLayer
        @test sens_layer !== base_layer  # Should be a new layer
        
        # The augmented layer should have more state variables
        base_prob = Corleone.get_problem(base_layer)
        sens_prob = Corleone.get_problem(sens_layer)
        @test length(sens_prob.u0) > length(base_prob.u0)
    end
end
