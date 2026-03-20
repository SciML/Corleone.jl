using Test
using CorleoneOED
using Corleone
using DifferentialEquations
using Random

@testset "Helper Functions" begin
    # Set up a simple ODE
    function odefn!(du, u, p, t)
        du[1] = -p.k * u[1]
    end
    
    u0 = [1.0]
    tspan = (0.0, 1.0)
    k = ControlParameter(0.5, (0.1, 2.0), :k)
    prob = ODEProblem(odefn!, u0, tspan, (k=k.value,))
    
    base_layer = SingleShootingLayer(prob, k; algorithm=Tsit5())
    
    @testset "create_oed_layer" begin
        # Test create_oed_layer
        oed_layer = create_oed_layer(
            base_layer,
            [:k],
            ContinuousMeasurement(k) => (vars, ps, t) -> vars[1]
        )
        
        @test oed_layer isa OEDLayerV2
        @test oed_layer.layer isa SingleShootingLayer
        @test oed_layer.symbolic_system isa SymbolicSystem
        
        # Test that it can be solved
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, oed_layer)
        initial_condition = InitialCondition([1.0])
        traj, st = oed_layer(initial_condition, ps, st)
        
        @test traj isa Trajectory
        @test length(traj.u) > 1
        
        # Test Fisher information extraction
        F = fisher_information(oed_layer, traj)
        @test F isa Matrix
        @test size(F) == (1, 1)
        @test F[1,1] >= 0  # Fisher info should be non-negative
    end
    
    @testset "augment_fisher" begin
        sys = get_symbolic_equations(base_layer)
        append_sensitivity!(sys, [:k])
        augment_fisher(sys, ContinuousMeasurement(k) => (vars, ps, t) -> vars[1])
        
        @test !isnothing(sys.fisher_continuous_vars)
        @test !isnothing(sys.fisher_continuous_eqs)
        
        # Create layer and test
        aug_layer = SingleShootingLayer(sys, base_layer)
        oed_layer = OEDLayerV2(sys, aug_layer)
        
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, oed_layer)
        initial_condition = InitialCondition([1.0])
        traj, st = oed_layer(initial_condition, ps, st)
        
        F = fisher_information(oed_layer, traj)
        @test F isa Matrix
        @test size(F) == (1, 1)
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
