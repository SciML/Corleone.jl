using Test
using CorleoneOED
using Corleone
using OrdinaryDiffEqTsit5
using Random
using LinearAlgebra
using LuxCore
using SymbolicIndexingInterface

@testset "Multi-Parameter OED Integration Tests" begin
    
    @testset "Two-parameter pharmacokinetics model" begin
        # A simple PK model: dC/dt = -k1*C + k2*D, dD/dt = k1*C - k2*D
        # C: central compartment, D: peripheral compartment
        # k1: distribution rate, k2: elimination rate
        function pk_model!(du, u, p, t)
            C, D = u
            k1, k2 = p[1], p[2]
            du[1] = -k1*C + k2*D
            du[2] = k1*C - k2*D
        end
        
        u0 = [10.0, 0.0]  # Initial dose in central compartment
        tspan = (0.0, 10.0)
        
        prob = ODEProblem{true}(
            ODEFunction(pk_model!, sys=SymbolCache([:C, :D], [:k1, :k2], :t)),
            u0, tspan, [0.3, 0.5]
        )
        
        k1 = ControlParameter(0.0:2.5:10.0, name=:k1)
        k2 = ControlParameter(0.0:2.5:10.0, name=:k2)
        base_layer = SingleShootingLayer(prob, k1, k2; algorithm=Tsit5())
        
        # Create OED layer with both parameters
        w = ControlParameter(0.0:2.5:10.0, name=:w)
        oed_layer = create_oed_layer(
            base_layer,
            [:k1, :k2],
            ContinuousMeasurement(w, (u, p, t) -> u[1])  # Measure central compartment
        )
        
        @test oed_layer isa OEDLayerV2
        
        # Solve
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, oed_layer)
        (F, traj), st = oed_layer(nothing, ps, st)
        
        # Extract Fisher information
        F_cont = fisher_information(oed_layer, traj)
        
        @test size(F_cont) == (2, 2)
        @test issymmetric(F_cont)
        @test all(eigvals(F_cont) .>= -1e-10)  # Should be positive semi-definite
        
        # Check sensitivity trajectories
        G_traj = sensitivities(oed_layer, traj)
        @test length(G_traj) == length(traj.u)
        @test size(G_traj[1]) == (2, 2)  # 2 states, 2 parameters
        
        # Initial sensitivities should be zero (states don't depend on params at t=0)
        @test all(abs.(G_traj[1]) .< 1e-10)
        
        # Final sensitivities should be non-zero
        @test !all(abs.(G_traj[end]) .< 1e-6)
        
        println("Two-parameter PK model Fisher matrix:")
        println(F_cont)
        println("Condition number: ", cond(F_cont))
    end
    
    @testset "Three-parameter SIR epidemic model" begin
        # SIR model: dS/dt = -β*S*I, dI/dt = β*S*I - γ*I, dR/dt = γ*I
        # β: infection rate, γ: recovery rate, N: total population
        function sir_model!(du, u, p, t)
            S, I, R = u
            β, γ = p[1], p[2]
            N = sum(u)
            du[1] = -β*S*I/N
            du[2] = β*S*I/N - γ*I
            du[3] = γ*I
        end
        
        # Start with 1% infected
        N = 1000.0
        u0 = [990.0, 10.0, 0.0]
        tspan = (0.0, 50.0)
        
        prob = ODEProblem{true}(
            ODEFunction(sir_model!, sys=SymbolCache([:S, :I, :R], [:β, :γ], :t)),
            u0, tspan, [0.5, 0.1]
        )
        
        β = ControlParameter(0.0:12.5:50.0, name=:β)
        γ = ControlParameter(0.0:12.5:50.0, name=:γ)
        base_layer = SingleShootingLayer(prob, β, γ; algorithm=Tsit5())
        
        # Create OED layer - measure infected population
        w = ControlParameter(0.0:12.5:50.0, name=:w)
        oed_layer = create_oed_layer(
            base_layer,
            [:β, :γ],
            ContinuousMeasurement(w, (u, p, t) -> u[2])  # Measure I(t)
        )
        
        # Solve
        rng = Random.default_rng()
        ps, st = LuxCore.setup(rng, oed_layer)
        (F, traj), st = oed_layer(nothing, ps, st)
        
        # Extract Fisher information
        F_cont = fisher_information(oed_layer, traj)
        
        @test size(F_cont) == (2, 2)
        @test issymmetric(F_cont)
        @test all(eigvals(F_cont) .>= -1e-10)
        
        println("\nSIR model Fisher matrix:")
        println(F_cont)
        
        # Check parameter identifiability via eigenvalue ratio
        eigs = sort(eigvals(F_cont), rev=true)
        if eigs[2] > 1e-6
            ratio = eigs[1] / eigs[2]
            println("Eigenvalue ratio (identifiability): ", ratio)
            @test ratio < 1e6  # Parameters should be relatively well-identified
        end
    end
    
    @testset "Comparing measurement strategies" begin
        # Simple decay with two parameters: dx/dt = -k1*x^k2
        function nonlinear_decay!(du, u, p, t)
            du[1] = -p[1] * u[1]^p[2]
        end
        
        u0 = [1.0]
        tspan = (0.0, 5.0)
        
        prob = ODEProblem{true}(
            ODEFunction(nonlinear_decay!, sys=SymbolCache([:x], [:k1, :k2], :t)),
            u0, tspan, [0.5, 1.0]
        )
        
        k1 = ControlParameter(0.0:1.25:5.0, name=:k1)
        k2 = ControlParameter(0.0:1.25:5.0, name=:k2)
        base_layer = SingleShootingLayer(prob, k1, k2; algorithm=Tsit5())
        
        # Strategy 1: Continuous measurement only
        w_cont = ControlParameter(0.0:1.25:5.0, name=:w_cont)
        oed_cont = create_oed_layer(
            base_layer,
            [:k1, :k2],
            ContinuousMeasurement(w_cont, (u, p, t) -> u[1])
        )
        
        rng = Random.default_rng()
        ps1, st1 = LuxCore.setup(rng, oed_cont)
        (F_cont_total, traj1), st1 = oed_cont(nothing, ps1, st1)
        F_cont = fisher_information(oed_cont, traj1)
        
        # Strategy 2: Discrete measurements only
        sys2 = get_symbolic_equations(base_layer)
        append_sensitivity!(sys2, [:k1, :k2])
        w_disc = ControlParameter(0.0:1.25:5.0, name=:w_disc)
        add_observed!(sys2, DiscreteMeasurement(w_disc, (u, p, t) -> u[1]))
        aug_layer2 = SingleShootingLayer(sys2, base_layer)
        oed_disc = OEDLayerV2(sys2, aug_layer2)
        
        ps2, st2 = LuxCore.setup(rng, oed_disc)
        (F_disc_total, traj2), st2 = oed_disc(nothing, ps2, st2)
        F_disc = discrete_fisher_information(oed_disc, traj2, ps2)
        
        # Strategy 3: Combined
        sys3 = get_symbolic_equations(base_layer)
        append_sensitivity!(sys3, [:k1, :k2])
        w_disc3 = ControlParameter(0.0:1.25:5.0, name=:w_disc3)
        w_cont3 = ControlParameter(0.0:1.25:5.0, name=:w_cont3)
        add_observed!(sys3,
            DiscreteMeasurement(w_disc3, (u, p, t) -> u[1]),
            ContinuousMeasurement(w_cont3, (u, p, t) -> u[1])
        )
        aug_layer3 = SingleShootingLayer(sys3, base_layer)
        oed_comb = OEDLayerV2(sys3, aug_layer3)
        
        ps3, st3 = LuxCore.setup(rng, oed_comb)
        (F_comb_total, traj3), st3 = oed_comb(nothing, ps3, st3)
        F_comb_cont = fisher_information(oed_comb, traj3)
        F_comb_disc = discrete_fisher_information(oed_comb, traj3, ps3)
        F_comb = F_comb_cont + F_comb_disc
        
        println("\nMeasurement strategy comparison:")
        println("Continuous only: ", tr(F_cont))
        println("Discrete only: ", tr(F_disc))
        println("Combined: ", tr(F_comb))
        
        # Combined should give more information (or equal within tolerance)
        @test tr(F_comb) >= tr(F_cont) - 1e-6
        @test tr(F_comb) >= tr(F_disc) - 1e-6
        
        # All should be positive semi-definite
        @test all(eigvals(F_cont) .>= -1e-10)
        @test all(eigvals(F_disc) .>= -1e-10)
        @test all(eigvals(F_comb) .>= -1e-10)
    end
end
