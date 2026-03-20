using Test
using CorleoneOED
using Corleone
using DifferentialEquations
using Random

@testset "Discrete Fisher Information" begin
    # Set up a simple exponential decay ODE: dx/dt = -k*x
    function odefn!(du, u, p, t)
        du[1] = -p.k * u[1]
    end
    
    u0 = [1.0]
    tspan = (0.0, 2.0)
    k = ControlParameter(0.5, (0.1, 2.0), :k)
    prob = ODEProblem(odefn!, u0, tspan, (k=k.value,))
    
    base_layer = SingleShootingLayer(prob, k; algorithm=Tsit5())
    
    @testset "Discrete measurement only" begin
        # Create OED layer with discrete measurement
        sys = get_symbolic_equations(base_layer)
        append_sensitivity!(sys, [:k])
        
        # Add discrete measurement: observe x directly
        disc_meas = DiscreteMeasurement(k) => (vars, ps, t) -> vars[1]
        add_observed!(sys, disc_meas)
        
        aug_layer = SingleShootingLayer(sys, base_layer)
        oed_layer = OEDLayerV2(sys, aug_layer)
        
        # Solve
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, oed_layer)
        initial_condition = InitialCondition([1.0])
        traj, st = oed_layer(initial_condition, ps, st)
        
        # Compute discrete Fisher at specific times
        meas_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        F_disc = discrete_fisher_information(oed_layer, traj, meas_times)
        
        @test F_disc isa Matrix
        @test size(F_disc) == (1, 1)
        @test F_disc[1,1] >= 0  # Fisher info should be non-negative
        @test isfinite(F_disc[1,1])
        
        println("Discrete Fisher info: ", F_disc[1,1])
    end
    
    @testset "Combined discrete and continuous" begin
        # Create OED layer with both measurement types
        sys = get_symbolic_equations(base_layer)
        append_sensitivity!(sys, [:k])
        
        disc_meas = DiscreteMeasurement(k) => (vars, ps, t) -> vars[1]
        cont_meas = ContinuousMeasurement(k) => (vars, ps, t) -> vars[1]
        add_observed!(sys, disc_meas, cont_meas)
        
        aug_layer = SingleShootingLayer(sys, base_layer)
        oed_layer = OEDLayerV2(sys, aug_layer)
        
        # Solve
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, oed_layer)
        initial_condition = InitialCondition([1.0])
        traj, st = oed_layer(initial_condition, ps, st)
        
        # Get both Fisher contributions
        F_cont = fisher_information(oed_layer, traj)
        meas_times = [0.0, 1.0, 2.0]
        F_disc = discrete_fisher_information(oed_layer, traj, meas_times)
        
        @test F_cont isa Matrix
        @test F_disc isa Matrix
        @test size(F_cont) == size(F_disc)
        
        # Total Fisher
        F_total = F_cont + F_disc
        @test F_total[1,1] >= F_cont[1,1]  # Discrete adds information
        @test F_total[1,1] >= F_disc[1,1]
        
        println("Continuous Fisher: ", F_cont[1,1])
        println("Discrete Fisher: ", F_disc[1,1])
        println("Total Fisher: ", F_total[1,1])
    end
    
    @testset "No discrete measurements returns zero" begin
        # Create OED layer with only continuous measurement
        sys = get_symbolic_equations(base_layer)
        append_sensitivity!(sys, [:k])
        cont_meas = ContinuousMeasurement(k) => (vars, ps, t) -> vars[1]
        add_observed!(sys, cont_meas)
        
        aug_layer = SingleShootingLayer(sys, base_layer)
        oed_layer = OEDLayerV2(sys, aug_layer)
        
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, oed_layer)
        initial_condition = InitialCondition([1.0])
        traj, st = oed_layer(initial_condition, ps, st)
        
        # Should return zero matrix
        F_disc = discrete_fisher_information(oed_layer, traj, [0.0, 1.0])
        @test F_disc == zeros(1, 1)
    end
end
