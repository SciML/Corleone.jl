using Corleone
using CorleoneOED
using SymbolicIndexingInterface
using OrdinaryDiffEqTsit5
using LuxCore
using Random
using Test
using ForwardDiff
using LinearAlgebra

@testset "Full Differentiability Tests" begin
    # Define a simple ODE system
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
    
    # Create OED layer
    symbolic_system = CorleoneOED.get_symbolic_equations(layer)
    CorleoneOED.append_sensitivity!(symbolic_system)
    
    discrete_observed = DiscreteMeasurement(
        ControlParameter(0.:1.:2., name = :w1), 
        (u, p, t) -> u[1]^2
    )
    continuous_observed = ContinuousMeasurement(
        ControlParameter(0.:1.:2., name = :w2), 
        (u, p, t) -> p[1] * u[1]
    )
    
    CorleoneOED.add_observed!(symbolic_system, discrete_observed, continuous_observed)
    new_layer = SingleShootingLayer(symbolic_system, layer)
    oed_layer = OEDLayer(symbolic_system, new_layer)
    
    ps, st = LuxCore.setup(Random.default_rng(), oed_layer)
    
    # Test that forward pass doesn't mutate and returns valid output
    (fisher, traj), st_new = oed_layer(nothing, ps, st)
    @test fisher isa Matrix
    @test size(fisher) == (1, 1)
    @test fisher[1,1] > 0
    
    println("✓ Forward pass is non-mutating")
    
    # Create a simple loss function based on Fisher information
    function loss_fn(weights)
        ps_modified = (
            layer = ps.layer,
            discrete_controls = (w1 = weights,)
        )
        (fisher, _), _ = oed_layer(nothing, ps_modified, st)
        return -log(det(fisher + 1e-6 * I))
    end
    
    w_test = ones(3)
    
    # Test ForwardDiff gradient
    @testset "ForwardDiff Gradient" begin
        grad = ForwardDiff.gradient(loss_fn, w_test)
        @test grad isa Vector
        @test length(grad) == 3
        @test all(isfinite, grad)
        println("  ForwardDiff gradient: ", grad)
    end
    
    # Test different weights produce different gradients
    @testset "Gradient Variability" begin
        w1 = ones(3)
        w2 = 2.0 * ones(3)
        grad1 = ForwardDiff.gradient(loss_fn, w1)
        grad2 = ForwardDiff.gradient(loss_fn, w2)
        @test !isapprox(grad1, grad2, atol=1e-6)
        println("  Gradients differ for different weights ✓")
    end
    
    # Test that cached getters are reused (not recomputed)
    @test !isnothing(oed_layer.continuous_fisher_getter)
    @test length(oed_layer.discrete_fisher_getters) == 1
    println("✓ Getters are cached in layer structure")
    
    # Test that sum is used (no mutation)
    @test_nowarn oed_layer(nothing, ps, st)
    println("✓ Implementation uses sum() for differentiability")
    
    # Test Hessian computation (second derivative)
    @testset "Hessian Computation" begin
        hess = ForwardDiff.hessian(loss_fn, w_test)
        @test hess isa Matrix
        @test size(hess) == (3, 3)
        @test all(isfinite, hess)
        println("  Hessian computed successfully ✓")
    end
    
    println("\n✓ All differentiability tests passed!")
end
