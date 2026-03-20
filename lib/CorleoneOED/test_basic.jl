#!/usr/bin/env julia
using Pkg
Pkg.activate(".")

println("Testing CorleoneOED basic functionality...")
println("="^70)

try
    using Corleone
    using CorleoneOED
    using SymbolicIndexingInterface
    using OrdinaryDiffEqTsit5
    using LuxCore
    using Random
    
    println("✓ All packages loaded successfully")
    
    # Simple 1D test
    println("\n1. Testing 1D exponential decay...")
    
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
    
    println("  ✓ Created base layer")
    
    # Extract symbolic system
    symbolic_system = CorleoneOED.get_symbolic_equations(layer)
    println("  ✓ Extracted symbolic equations")
    
    # Add sensitivities
    CorleoneOED.append_sensitivity!(symbolic_system)
    println("  ✓ Added sensitivities")
    
    # Add measurements
    w1 = ControlParameter(0.0:0.1:2.0, name=:w1)
    w2 = ControlParameter(0.0:0.1:2.0, name=:w2)
    
    discrete_obs = DiscreteMeasurement(w1, (u, p, t) -> u[1]^2)
    continuous_obs = ContinuousMeasurement(w2, (u, p, t) -> p[1] * u[1])
    
    CorleoneOED.add_observed!(symbolic_system, discrete_obs, continuous_obs)
    println("  ✓ Added measurements")
    
    # Create augmented layer
    new_layer = SingleShootingLayer(symbolic_system, layer)
    println("  ✓ Created augmented layer")
    
    # Create OED layer
    oed_layer = OEDLayer(symbolic_system, new_layer)
    println("  ✓ Created OED layer")
    
    # Setup and solve
    rng = Random.default_rng()
    ps, st = LuxCore.setup(rng, oed_layer)
    println("  ✓ Initialized parameters and state")
    println("    Parameters: ", keys(ps))
    
    # Call the layer
    println("  → Solving ODE system...")
    (fisher, trajectory), st_new = oed_layer(nothing, ps, st)
    
    println("  ✓ Solution computed")
    println("    Time points: ", length(trajectory.t))
    println("    Fisher matrix: ", size(fisher))
    println("    Fisher value: ", fisher[1,1])
    
    @assert fisher isa Matrix
    @assert size(fisher) == (1, 1)
    @assert fisher[1,1] > 0
    @assert trajectory isa Trajectory
    
    println("\n✅ All basic functionality tests passed!")
    println("="^70)
    
    exit(0)
    
catch e
    println("\n❌ Error occurred:")
    println(e)
    for (exc, bt) in Base.catch_stack()
        showerror(stdout, exc, bt)
        println()
    end
    exit(1)
end
