using Test
using CorleoneOED
using Corleone
using OrdinaryDiffEqTsit5
using Random
using LuxCore
using SymbolicIndexingInterface

@testset "Discrete Fisher Information" begin
    # Set up a simple exponential decay ODE: dx/dt = -k*x
    function odefn!(du, u, p, t)
        du[1] = -p[1] * u[1]
    end
    
    u0 = [1.0]
    tspan = (0.0, 2.0)
    
    prob = ODEProblem{true}(
        ODEFunction(odefn!, sys=SymbolCache([:x], [:k], :t)),
        u0, tspan, [0.5]
    )
    
    k = ControlParameter(0.0:0.5:2.0, name=:k)
    base_layer = SingleShootingLayer(prob, k; algorithm=Tsit5())
    
    @testset "Discrete measurement only" begin
        # Create OED layer with discrete measurement
        sys = get_symbolic_equations(base_layer)
        append_sensitivity!(sys)
        
        # Add discrete measurement: observe x directly
        w1 = ControlParameter(0.0:0.5:2.0, name=:w1)
        disc_meas = DiscreteMeasurement(w1, (u, p, t) -> u[1])
        add_observed!(sys, disc_meas)
        
        aug_layer = SingleShootingLayer(sys, base_layer)
        oed_layer = OEDLayer(sys, aug_layer)
        
        # Solve
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, oed_layer)
        (F, traj), st = oed_layer(nothing, ps, st)
        
        # Check discrete Fisher
        F_disc = discrete_fisher_information(oed_layer, traj, ps)
        
        @test F_disc isa Matrix
        @test size(F_disc) == (1, 1)
        @test F_disc[1,1] >= 0  # Fisher info should be non-negative
        @test isfinite(F_disc[1,1])
        
        println("Discrete Fisher info: ", F_disc[1,1])
    end
    
    @testset "Combined discrete and continuous" begin
        # Create OED layer with both measurement types
        sys = get_symbolic_equations(base_layer)
        append_sensitivity!(sys)
        
        w1 = ControlParameter(0.0:0.5:2.0, name=:w1)
        w2 = ControlParameter(0.0:0.5:2.0, name=:w2)
        disc_meas = DiscreteMeasurement(w1, (u, p, t) -> u[1])
        cont_meas = ContinuousMeasurement(w2, (u, p, t) -> u[1])
        add_observed!(sys, disc_meas, cont_meas)
        
        aug_layer = SingleShootingLayer(sys, base_layer)
        oed_layer = OEDLayer(sys, aug_layer)
        
        # Solve
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, oed_layer)
        (F_total, traj), st = oed_layer(nothing, ps, st)
        
        # Get both Fisher contributions
        F_cont = fisher_information(oed_layer, traj)
        F_disc = discrete_fisher_information(oed_layer, traj, ps)
        
        @test F_cont isa Matrix
        @test F_disc isa Matrix
        @test size(F_cont) == size(F_disc)
        
        # Total Fisher
        @test isapprox(F_total, F_cont + F_disc, atol=1e-10)
        @test F_total[1,1] >= F_cont[1,1] - 1e-10  # Discrete adds information
        @test F_total[1,1] >= F_disc[1,1] - 1e-10
        
        println("Continuous Fisher: ", F_cont[1,1])
        println("Discrete Fisher: ", F_disc[1,1])
        println("Total Fisher: ", F_total[1,1])
    end
    
    @testset "No discrete measurements returns zero" begin
        # Create OED layer with only continuous measurement
        sys = get_symbolic_equations(base_layer)
        append_sensitivity!(sys)
        
        w2 = ControlParameter(0.0:0.5:2.0, name=:w2)
        cont_meas = ContinuousMeasurement(w2, (u, p, t) -> u[1])
        add_observed!(sys, cont_meas)
        
        aug_layer = SingleShootingLayer(sys, base_layer)
        oed_layer = OEDLayer(sys, aug_layer)
        
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, oed_layer)
        (F, traj), st = oed_layer(nothing, ps, st)
        
        # Should return zero matrix (no discrete measurements)
        F_disc = discrete_fisher_information(oed_layer, traj, ps)
        @test F_disc == zeros(1, 1)
    end
end
