using Corleone
using CorleoneOED
using SymbolicIndexingInterface
using OrdinaryDiffEqTsit5
using LuxCore
using Random
using Test

@testset "Example Corrected Test" begin
    # Define a Problem using Corleone 
    function f(du, u, p, t)
        du[1] = -p[1] * u[1]
    end
    u0 = [1.0]
    tspan = (0.0, 2.0)
    p = [0.5]
    
    prob = ODEProblem{true}(
        ODEFunction(f, sys=SymbolCache([:x], [:p], :t)), u0, tspan, p)
    control = Corleone.ControlParameter(0.0:0.5:2.0, name=:p)
    layer = Corleone.SingleShootingLayer(prob, control; algorithm=Tsit5())
    
    # Extract symbolic system
    symbolic_system = CorleoneOED.get_symbolic_equations(layer)
    @test symbolic_system isa SymbolicSystem
    
    # Append forward sensitivity (all parameters by default)
    CorleoneOED.append_sensitivity!(symbolic_system)
    @test !isnothing(symbolic_system.sensitivities)
    @test !isnothing(symbolic_system.sensitivity_equations)
    
    # Add discrete and continuous measurements
    discrete_observed = DiscreteMeasurement(
        ControlParameter(0.:1.:2., name = :w1), 
        (u, p, t) -> u[1]^2
    )
    continuous_observed = ContinuousMeasurement(
        ControlParameter(0.:1.:2., name = :w2), 
        (u, p, t) -> p[1] * u[1]
    )
    
    CorleoneOED.add_observed!(symbolic_system, discrete_observed, continuous_observed)
    @test length(symbolic_system.discrete_measurements) == 1
    @test length(symbolic_system.continuous_measurements) == 1
    @test length(symbolic_system.discrete_measurement_controls) == 1  # w1
    @test length(symbolic_system.continuous_measurement_controls) == 1  # w2
    
    # Check that continuous Fisher equations were created
    @test !isnothing(symbolic_system.fisher_continuous_vars)
    @test !isnothing(symbolic_system.fisher_continuous_eqs)
    
    # Create augmented layer
    new_layer = SingleShootingLayer(symbolic_system, layer)
    @test new_layer isa Corleone.SingleShootingLayer
    
    # Check that the new layer contains both original control and measurement controls
    ps, st = LuxCore.setup(Random.default_rng(), new_layer)
    @test haskey(ps.controls, :p)  # Original control
    @test haskey(ps.controls, :w2)  # Continuous measurement control (in ODE)
    
    # Wrap in OEDLayer
    oed_layer = OEDLayer(symbolic_system, new_layer)
    @test oed_layer isa OEDLayer
    
    # Check that OED layer also has both controls
    ps_oed, st_oed = LuxCore.setup(Random.default_rng(), oed_layer)
    @test haskey(ps_oed.layer.controls, :p)  # Original control
    @test haskey(ps_oed.layer.controls, :w2)  # Continuous measurement control
    @test haskey(ps_oed.discrete_controls, :w1)  # Discrete measurement control
    
    # Call the layer - should return (fisher, trajectory) tuple
    result, st_new = oed_layer(nothing, ps_oed, st_oed)
    @test result isa Tuple
    @test length(result) == 2
    
    fisher, trajectory = result
    @test fisher isa Matrix
    @test size(fisher) == (1, 1)  # 1 parameter (p)
    @test fisher[1,1] >= 0  # Fisher info should be non-negative
    @test trajectory isa Trajectory
    
    println("✓ Example corrected test passed!")
    println("  Fisher information: ", fisher[1,1])
end
