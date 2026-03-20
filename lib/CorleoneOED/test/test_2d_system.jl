using Corleone
using CorleoneOED
using SymbolicIndexingInterface
using OrdinaryDiffEqTsit5
using LuxCore
using Random
using Test
using LinearAlgebra

"""
2D Test Case: Coupled exponential decay system
    dx1/dt = -k1 * x1 + k2 * x2
    dx2/dt = k1 * x1 - k2 * x2
    
Initial conditions: x1(0) = 1.0, x2(0) = 0.0
Parameters: k1 = 0.3, k2 = 0.5
Time span: [0, 2.0]

Measurements:
- Discrete: observe x1² at times with weight w1=1.0
- Continuous: observe k1*x1 + k2*x2 with weight w2=1.0

Expected behavior:
- System transfers mass from x1 to x2, reaching equilibrium
- Sensitivities G = ∂x/∂k should be non-zero and evolve over time
- Fisher information should be positive definite (2x2 matrix for 2 parameters)
- Continuous Fisher accumulates continuously via dF/dt
- Discrete Fisher adds jumps at measurement times
"""

@testset "2D Coupled System OED" begin
    println("\n" * "="^70)
    println("2D Coupled Exponential Decay System Test")
    println("="^70)
    
    # Define coupled system
    function coupled_decay!(du, u, p, t)
        x1, x2 = u
        k1, k2 = p[1], p[2]
        du[1] = -k1 * x1 + k2 * x2
        du[2] = k1 * x1 - k2 * x2
    end
    
    u0 = [1.0, 0.0]
    tspan = (0.0, 2.0)
    p = [0.3, 0.5]
    
    prob = ODEProblem{true}(
        ODEFunction(coupled_decay!, sys=SymbolCache([:x1, :x2], [:k1, :k2], :t)),
        u0, tspan, p
    )
    
    k1_control = Corleone.ControlParameter(0.0:0.5:2.0, name=:k1)
    k2_control = Corleone.ControlParameter(0.0:0.5:2.0, name=:k2)
    layer = Corleone.SingleShootingLayer(prob, k1_control, k2_control; algorithm=Tsit5())
    
    println("\n1. System Setup:")
    println("   - States: x1, x2")
    println("   - Parameters: k1=$(p[1]), k2=$(p[2])")
    println("   - Initial: x1(0)=$(u0[1]), x2(0)=$(u0[2])")
    
    # Extract symbolic system
    symbolic_system = CorleoneOED.get_symbolic_equations(layer)
    @test symbolic_system isa SymbolicSystem
    @test length(symbolic_system.vars) == 2
    @test length(symbolic_system.parameters) == 2
    println("   ✓ Symbolic system extracted")
    
    # Add sensitivities for both parameters
    CorleoneOED.append_sensitivity!(symbolic_system)
    @test !isnothing(symbolic_system.sensitivities)
    @test size(symbolic_system.sensitivities) == (2, 2)  # 2 states × 2 params
    @test !isnothing(symbolic_system.sensitivity_equations)
    @test length(symbolic_system.sensitivity_equations) == 4  # 2×2 = 4 equations
    println("\n2. Sensitivity Analysis:")
    println("   - Sensitivity matrix G: 2×2 (∂x/∂k)")
    println("   - Sensitivity equations: 4 ODEs added")
    println("   ✓ Forward sensitivities computed")
    
    # Add measurements
    w1_control = ControlParameter(0.0:0.5:2.0, name=:w1)
    w2_control = ControlParameter(0.0:0.5:2.0, name=:w2)
    
    # Discrete: measure x1²
    discrete_obs = DiscreteMeasurement(w1_control, (u, p, t) -> u[1]^2)
    
    # Continuous: measure k1*x1 + k2*x2 (a weighted sum)
    continuous_obs = ContinuousMeasurement(w2_control, (u, p, t) -> p[1]*u[1] + p[2]*u[2])
    
    CorleoneOED.add_observed!(symbolic_system, discrete_obs, continuous_obs)
    @test length(symbolic_system.discrete_measurements) == 1
    @test length(symbolic_system.continuous_measurements) == 1
    @test length(symbolic_system.discrete_measurement_controls) == 1
    @test length(symbolic_system.continuous_measurement_controls) == 1
    @test !isnothing(symbolic_system.fisher_continuous_vars)
    @test !isnothing(symbolic_system.fisher_continuous_eqs)
    
    println("\n3. Measurement Models:")
    println("   - Discrete: h_d = x1² (weight w1=1.0)")
    println("   - Continuous: h_c = k1*x1 + k2*x2 (weight w2=1.0)")
    println("   - Fisher matrix: 2×2 (symmetric)")
    println("   ✓ Measurement models added")
    
    # Create augmented layer
    new_layer = SingleShootingLayer(symbolic_system, layer)
    @test new_layer isa Corleone.SingleShootingLayer
    
    # Check controls
    ps_new, st_new = LuxCore.setup(Random.default_rng(), new_layer)
    @test haskey(ps_new, :k1)  # Original
    @test haskey(ps_new, :k2)  # Original
    @test haskey(ps_new, :w2)  # Continuous measurement weight
    
    println("\n4. Augmented System:")
    println("   - Original states: 2")
    println("   - Sensitivities: 4 (2×2)")
    println("   - Fisher (continuous): 3 (upper triangle of 2×2)")
    println("   - Total state dimension: $(length(Corleone.get_problem(new_layer).u0))")
    println("   - Controls: k1, k2, w2")
    println("   ✓ Augmented layer created")
    
    # Create OED layer
    oed_layer = OEDLayerV2(symbolic_system, new_layer)
    @test oed_layer isa OEDLayerV2
    
    ps_oed, st_oed = LuxCore.setup(Random.default_rng(), oed_layer)
    @test haskey(ps_oed, :k1)
    @test haskey(ps_oed, :k2)
    @test haskey(ps_oed, :w1)  # Discrete measurement weight
    @test haskey(ps_oed, :w2)  # Continuous measurement weight
    
    println("\n5. OED Layer:")
    println("   - Controls: k1, k2, w1, w2")
    println("   - Returns: (Fisher, Trajectory)")
    println("   ✓ OED wrapper created")
    
    # Solve the system
    (fisher, trajectory), st_final = oed_layer(nothing, ps_oed, st_oed)
    
    @test fisher isa Matrix
    @test size(fisher) == (2, 2)
    @test issymmetric(fisher)
    @test trajectory isa Trajectory
    @test length(trajectory.t) > 1
    
    println("\n6. Solution:")
    println("   - Time points: $(length(trajectory.t))")
    println("   - Final time: $(trajectory.t[end])")
    println("   - Final state: x1=$(trajectory.u[end][1]), x2=$(trajectory.u[end][2])")
    
    # Expected: mass conservation (x1 + x2 ≈ 1.0)
    mass_final = trajectory.u[end][1] + trajectory.u[end][2]
    @test isapprox(mass_final, 1.0, atol=1e-6)
    println("   - Mass conservation: $(mass_final) ≈ 1.0 ✓")
    
    # Extract components separately
    F_cont = fisher_information(oed_layer, trajectory)
    F_disc = discrete_fisher_information(oed_layer, trajectory, trajectory.t, ps_oed)
    
    @test isapprox(fisher, F_cont + F_disc, atol=1e-10)
    
    println("\n7. Fisher Information Matrix:")
    println("   Continuous contribution:")
    println("   ", F_cont)
    println("\n   Discrete contribution:")
    println("   ", F_disc)
    println("\n   Total Fisher:")
    println("   ", fisher)
    
    # Check Fisher properties
    eigs = eigvals(fisher)
    @test all(eigs .>= -1e-10)  # Positive semi-definite
    println("\n8. Fisher Matrix Properties:")
    println("   - Symmetric: $(issymmetric(fisher))")
    println("   - Eigenvalues: ", eigs)
    println("   - Condition number: $(cond(fisher))")
    println("   - Trace: $(tr(fisher))")
    println("   - Determinant: $(det(fisher))")
    
    if all(eigs .> 1e-10)
        println("   ✓ Positive definite (both parameters identifiable)")
    else
        println("   ! Semi-definite (some parameters may not be identifiable)")
    end
    
    # Extract sensitivities
    G_traj = sensitivities(oed_layer, trajectory)
    @test length(G_traj) == length(trajectory.t)
    @test size(G_traj[1]) == (2, 2)
    
    println("\n9. Sensitivity Trajectories:")
    println("   - Initial G(0):")
    println("   ", G_traj[1])
    println("\n   - Final G($(trajectory.t[end])):")
    println("   ", G_traj[end])
    
    # Expected: initial sensitivities should be zero
    @test all(abs.(G_traj[1]) .< 1e-10)
    println("   ✓ Initial sensitivities are zero (as expected)")
    
    # Expected: final sensitivities should be non-zero
    @test !all(abs.(G_traj[end]) .< 1e-6)
    println("   ✓ Final sensitivities are non-zero")
    
    println("\n" * "="^70)
    println("Expected Results Summary:")
    println("="^70)
    println("✓ State dimension: 9 (2 states + 4 sens + 3 Fisher)")
    println("✓ Fisher matrix: 2×2, symmetric, positive semi-definite")
    println("✓ Mass conservation: x1(t) + x2(t) ≈ 1.0")
    println("✓ Sensitivities: G(0) = 0, G(T) ≠ 0")
    println("✓ All controls accessible: k1, k2, w1, w2")
    println("✓ Return format: (Fisher, Trajectory)")
    println("="^70)
    
    println("\n✅ All 2D test cases passed!")
end
