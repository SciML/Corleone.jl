# Test for MTK integration - mirrors lotka_oc.jl structure
# Note: MTK only supports ForwardDiff for automatic differentiation

using Test
using Corleone
using Corleone: get_lower_bound, get_upper_bound
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using ModelingToolkit.Symbolics: Integral, operation, unwrap
using OrdinaryDiffEqTsit5
using Random
using LuxCore
using SymbolicIndexingInterface
using ComponentArrays
using Optimization, OptimizationMOI, Ipopt
using SciMLSensitivity

rng = Random.default_rng()

# Define Lotka-Volterra optimal control problem using MTK
@variables begin
    x(t) = 0.5, [tunable = false, bounds = (0.0, Inf)]
    y(t) = 0.7, [tunable = false, bounds = (0.0, Inf)]
    c(t) = 0.0, [tunable = false, bounds = (-Inf, Inf)]  # Cost variable
end

# Control input with bounds
@variables begin
    u(t) = 0.0, [input = true, bounds = (0.0, 1.0)]
end

# Parameters with bounds for optimization
@parameters begin
    α = 1.0, [bounds = (0.0, Inf)]  # Tunable parameter
    β = 1.0, [bounds = (0.0, Inf)]  # Tunable parameter
end

# Lotka-Volterra dynamics with control
# dx/dt = x - β*x*y - 0.4*u*x
# dy/dt = -y + α*x*y - 0.2*u*y
# Cost: (x - 1)^2 + (y - 1)^2 integrated over time
eqs = [
    D(x) ~ x - β * x * y - 0.4 * u * x,
    D(y) ~ -y + α * x * y - 0.2 * u * y,
    D(c) ~ (x - 1.0)^2 + (y - 1.0)^2,
]

@named lotka_system = ODESystem(eqs, t)

# Time grid for control discretization (same as lotka_oc.jl)
cgrid = collect(0.0:0.1:11.9)
N = length(cgrid)

@testset "MTK Single Shooting" begin
    # Create SingleShootingLayer with MTK system
    layer = SingleShootingLayer(
        lotka_system,
        [],  # No initial condition overrides
        u => cgrid;
        algorithm = Tsit5(),
        tspan = (0.0, 12.0),
        quadrature_indices = [3]  # Index of cost variable
    )
    
    ps, st = LuxCore.setup(rng, layer)
    sol, _ = layer(nothing, ps, st)
    
    # Time access
    @test sol.t == getsym(sol, :t)(sol)
    
    # Parameter access via symbolic vs symbol should match
    @test all(sol.ps[:α] .== getsym(sol, α)(sol))
    @test all(sol.ps[:β] .== getsym(sol, β)(sol))
    
    # Control values
    @test ps.controls.u == sol.ps[:u][1:(end - 1)]
    
    # State values via getsym - check lengths match
    @test length(getsym(sol, x)(sol)) == length(sol.t)
    @test length(getsym(sol, y)(sol)) == length(sol.t)
    @test length(getsym(sol, c)(sol)) == length(sol.t)
    
    # Type inference
    @test_nowarn @inferred first(layer(nothing, ps, st))
    @test allunique(sol.t)
    
    # Parameter length: N control values for u + 2 parameters (α, β)
    @test LuxCore.parameterlength(layer) == N + 2
    
    # Test bounds - u bounds: [0, 1]
    lb = get_lower_bound(layer)
    ub = get_upper_bound(layer)
    
    @test all(lb.controls.u .>= 0.0 - 1.0e-6)
    @test all(ub.controls.u .<= 1.0 + 1.0e-6)
    
    # α and β bounds from MTK definition: [0, Inf]
    @test lb.controls.α[1] >= 0.0 - 1.0e-6
    @test isinf(ub.controls.α[1]) && ub.controls.α[1] > 0
    @test lb.controls.β[1] >= 0.0 - 1.0e-6
    @test isinf(ub.controls.β[1]) && ub.controls.β[1] > 0
    
    # Test trajectory values are reasonable
    x_vals = getsym(sol, x)(sol)
    y_vals = getsym(sol, y)(sol)
    @test all(x_vals .>= -1.0e-6)
    @test all(y_vals .>= -1.0e-6)
end

@testset "MTK Control Bounds" begin
    # Test that bounds from MTK variables are correctly propagated
    layer = SingleShootingLayer(
        lotka_system,
        [],
        u => cgrid;
        algorithm = Tsit5(),
        tspan = (0.0, 1.0)
    )
    
    ps, st = LuxCore.setup(rng, layer)
    
    # Get bounds
    lb = get_lower_bound(layer)
    ub = get_upper_bound(layer)
    
    # Control u should have bounds [0, 1]
    u_lb = lb.controls.u
    u_ub = ub.controls.u
    
    @test all(u_lb .>= 0.0 - 1.0e-6)
    @test all(u_ub .<= 1.0 + 1.0e-6)
    
    # Parameters should have bounds from MTK definition
    @test :α in keys(lb.controls)
    @test :β in keys(lb.controls)
end

@testset "MTK Parameter Access Patterns" begin
    layer = SingleShootingLayer(
        lotka_system,
        [],
        u => 0.0:0.1:1.0;
        algorithm = Tsit5(),
        tspan = (0.0, 1.0)
    )
    
    ps, st = LuxCore.setup(rng, layer)
    traj, _ = layer(nothing, ps, st)
    
    # Test getp for control parameters
    α_getter = getp(traj, α)
    β_getter = getp(traj, β)
    u_getter = getp(traj, u)
    
    @test α_getter(traj) isa Vector
    @test β_getter(traj) isa Vector
    @test u_getter(traj) isa Vector
    
    # Test sizes
    @test length(α_getter(traj)) == length(traj.t)
    @test length(u_getter(traj)) == length(traj.t)
end

@testset "MTK DynamicOptimizationLayer" begin
    # Test basic DynamicOptimizationLayer construction and evaluation
    layer = SingleShootingLayer(
        lotka_system,
        [],
        u => cgrid;
        algorithm = Tsit5(),
        tspan = (0.0, 1.0),
        quadrature_indices = [3]
    )
    
    optlayer = DynamicOptimizationLayer(layer, :(c(1.0)))
    ps, st = LuxCore.setup(rng, optlayer)
    
    # Test that we can evaluate the layer
    result = optlayer(nothing, ps, st)
    obj_val = result[1]
    
    # Objective should be positive (cost)
    @test obj_val > 0
    
    # Test layer has correct structure
    @test optlayer isa DynamicOptimizationLayer
end

@testset "MTK Quadrature" begin
    # Test that quadrature variables work correctly
    layer = SingleShootingLayer(
        lotka_system,
        [],
        u => 0.0:0.1:1.0;
        algorithm = Tsit5(),
        tspan = (0.0, 1.0),
        quadrature_indices = [3]  # c is the cost variable
    )
    
    ps, st = LuxCore.setup(rng, layer)
    traj, _ = layer(nothing, ps, st)
    
    # c should accumulate the cost over time
    c_vals = getsym(traj, c)(traj)
    
    # Cost should be non-negative and increase over time
    @test all(c_vals .>= -1.0e-6)
    @test c_vals[end] >= c_vals[1]  # Cost integral should grow
end

@testset "MTK Single Shooting IPOPT Optimization" begin
    # Full optimization test with IPOPT - similar to lotka_oc.jl lines 91-110
    # MTK only supports ForwardDiff for AD
    
    layer = SingleShootingLayer(
        lotka_system,
        [],  # No initial condition overrides
        u => cgrid;
        algorithm = Tsit5(),
        tspan = (0.0, 12.0),
        quadrature_indices = [3]  # Index of cost variable
    )
    
    ps, st = LuxCore.setup(rng, layer)
    sol, _ = layer(nothing, ps, st)
    
    # Verify basic layer functionality first
    @test sol.t == getsym(sol, :t)(sol)
    @test all(sol.ps[:α] .== getsym(sol, α)(sol))
    @test all(sol.ps[:β] .== getsym(sol, β)(sol))
    
    # Test bounds before optimization
    lb = get_lower_bound(layer)
    ub = get_upper_bound(layer)
    @test all(lb.controls.u .>= 0.0 - 1.0e-6)
    @test all(ub.controls.u .<= 1.0 + 1.0e-6)
    
    # Run optimization with AutoForwardDiff (MTK supports only ForwardDiff)
    for AD in (AutoForwardDiff(),)
        layer = remake(layer, sensealg = SciMLBase.NoAD())
        optlayer = DynamicOptimizationLayer(layer, :(c(12.0)))
        ps, st = LuxCore.setup(rng, optlayer)
        
        # Test type inference
        @test_nowarn @inferred first(optlayer(nothing, ps, st))
        
        # Create optimization problem
        optprob = OptimizationProblem(optlayer, AD, vectorizer = Val(:ComponentArrays))
        p = ComponentArray(ps)
        
        # Test initial objective value (should match lotka_oc.jl: ~6.062277)
        @test isapprox(optprob.f(optprob.u0, optprob.p), 6.062277454291031, atol = 1.0e-4)
        
        # Test bounds
        @test all(optprob.ub .== 1.0)
        @test all(optprob.lb .== 0.0)
        
        # Solve with IPOPT
        sol = solve(
            optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
            hessian_approximation = "limited-memory"
        )
        
        # Verify successful optimization
        @test SciMLBase.successful_retcode(sol)
        
        # Test final objective (should match lotka_oc.jl: ~1.344336)
        @test isapprox(sol.objective, 1.344336, atol = 1.0e-4)
        
        # Verify optimized parameters
        p_opt = sol.u .+ zero(p)
        @test isempty(p_opt.initial_conditions)
        @test length(p_opt.controls.u) == N
    end
end

@testset "MTK Multiple Shooting IPOPT Optimization" begin
    # Multiple shooting optimization test - similar to lotka_oc.jl lines 113-143
    
    layer = SingleShootingLayer(
        lotka_system,
        [];
        bounds_ic = (t0) -> (zeros(3), fill(Inf, 3)),
        algorithm = Tsit5(),
        tspan = (0.0, 12.0),
        quadrature_indices = [3]
    )
    
    ms_layer = MultipleShootingLayer(layer, 0.0, 3.0, 6.0, 9.0)
    ps, st = LuxCore.setup(rng, ms_layer)
    traj, _ = ms_layer(nothing, ps, st)
    
    # Test shooting constraints count
    @test Corleone.get_number_of_shooting_constraints(ms_layer) == 6
    
    # Run optimization with AutoForwardDiff
    for AD in (AutoForwardDiff(),)
        ms_layer = remake(ms_layer, sensealg = SciMLBase.NoAD())
        optlayer = DynamicOptimizationLayer(ms_layer, :(c(12.0)))
        
        # Test constraint bounds
        @test length(optlayer.lcons) == length(optlayer.ucons) == 6
        @test optlayer.lcons == optlayer.ucons
        
        ps, st = LuxCore.setup(rng, optlayer)
        
        # Test type inference
        objectiveval = @inferred first(optlayer(nothing, ps, st))
        
        # Test initial objective (should match lotka_oc.jl: ~4.966904)
        @test isapprox(objectiveval, 4.9669040432037574, atol = 1.0e-4)
        
        # Test constraint evaluation
        res = zeros(6)
        @inferred first(optlayer(res, ps, st))
        @test isapprox(res, [-1.3757549609694821, -0.2235735751355118, -1.375754960969481, -0.22357357513551102, -1.3757549609694824, -0.2235735751355129], atol = 1.0e-4)
        
        # Create and solve optimization problem
        optprob = OptimizationProblem(optlayer, AutoForwardDiff(), vectorizer = Val(:ComponentArrays))
        sol = solve(
            optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
            hessian_approximation = "limited-memory"
        )
        
        # Verify successful optimization
        @test SciMLBase.successful_retcode(sol)
        
        # Test final objective (should match lotka_oc.jl: ~1.344336)
        @test isapprox(sol.objective, 1.344336, atol = 1.0e-4)
    end
end

# ============================================================================
# Symbolic Interface Tests - Using DynamicOptimizationLayer(sys, ...) with Integrals
# Note: This interface uses MTK systems with symbolic expressions for objectives/constraints
# ============================================================================

@testset "MTK Symbolic Interface - Terminal Cost" begin
    # Test DynamicOptimizationLayer(sys, defaults, controls, expr) interface
    # Uses symbolic expression c(t_final) as terminal cost (no Integral)
    # Note: This test uses the standard DynamicOptimizationLayer(layer, :(expr)) approach
    # because the MTK symbolic interface for terminal costs requires special handling
    # of time points. For Integral expressions, the symbolic interface works directly.
    
    @variables begin
        x_sym(t) = 0.5, [tunable = false]
        y_sym(t) = 0.7, [tunable = false]
        c_sym(t) = 0.0, [tunable = false]  # Cost accumulator
    end
    @variables begin
        u_sym(t) = 0.0, [input = true, bounds = (0.0, 1.0)]
    end
    
    # Same parameters as main test (tunable with same bounds)
    @parameters begin
        α_sym = 1.0, [bounds = (0.0, Inf)]
        β_sym = 1.0, [bounds = (0.0, Inf)]
    end
    
    # Same dynamics as main test
    eqs_sym = [
        D(x_sym) ~ x_sym - β_sym * x_sym * y_sym - 0.4 * u_sym * x_sym,
        D(y_sym) ~ -y_sym + α_sym * x_sym * y_sym - 0.2 * u_sym * y_sym,
        D(c_sym) ~ (x_sym - 1.0)^2 + (y_sym - 1.0)^2,
    ]
    
    @named lotka_sym = ODESystem(eqs_sym, t)
    
    # Create SingleShootingLayer first, then wrap with DynamicOptimizationLayer
    # This matches the pattern used in MTK DynamicOptimizationLayer test
    layer = SingleShootingLayer(
        lotka_sym,
        [],  # No defaults overrides
        u_sym => cgrid;
        algorithm = Tsit5(),
        tspan = (0.0, 12.0),
        quadrature_indices = [3]  # Index of c_sym
    )
    
    # Create DynamicOptimizationLayer with quoted expression (same as non-MTK interface)
    optlayer = DynamicOptimizationLayer(layer, :(c_sym(12.0)))
    
    ps, st = LuxCore.setup(rng, optlayer)
    
    # Test evaluation
    result = @inferred first(optlayer(nothing, ps, st))
    @test result > 0  # Should be a positive cost
    
    # Test optimization problem construction
    optprob = OptimizationProblem(optlayer, AutoForwardDiff(), vectorizer = Val(:ComponentArrays))
    @test optprob isa OptimizationProblem
    
    # Test initial objective (should be positive)
    @test optprob.f(optprob.u0, optprob.p) > 0
end

@testset "MTK Symbolic Interface - Lagrangian Cost with Integral" begin
    # Test using Symbolics.Integral for Lagrangian cost term
    # ∫₀ᴰ ((x-1)² + (y-1)²) dt
    # Uses SAME dynamics as main test for comparable numerical results
    
    @variables begin
        x_int(t) = 0.5, [tunable = false]
        y_int(t) = 0.7, [tunable = false]
    end
    @variables begin
        u_int(t) = 0.0, [input = true, bounds = (0.0, 1.0)]
    end
    
    # Same parameters as main test (tunable with same bounds)
    @parameters begin
        α_int = 1.0, [bounds = (0.0, Inf)]
        β_int = 1.0, [bounds = (0.0, Inf)]
    end
    
    # Same dynamics as main test (without explicit c state since Integral adds it)
    eqs_int = [
        D(x_int) ~ x_int - β_int * x_int * y_int - 0.4 * u_int * x_int,
        D(y_int) ~ -y_int + α_int * x_int * y_int - 0.2 * u_int * y_int,
    ]
    
    # Define Lagrangian cost using Symbolics.Integral
    lagrangian = Integral(t in (0.0, 12.0))(
        (x_int - 1.0)^2 + (y_int - 1.0)^2
    )
    
    @named lotka_integral = ODESystem(eqs_int, t)
    
    # Create DynamicOptimizationLayer with Integral expression
    optlayer = DynamicOptimizationLayer(
        lotka_integral,
        [],
        u_int => cgrid,  # Use variable symbol, not call
        lagrangian;  # Pass Integral expression directly
        algorithm = Tsit5()
    )
    
    ps, st = LuxCore.setup(rng, optlayer)
    
    # Test evaluation
    result = @inferred first(optlayer(nothing, ps, st))
    @test result > 0  # Should be positive cost
    
    # Test optimization problem construction
    optprob = OptimizationProblem(optlayer, AutoForwardDiff(), vectorizer = Val(:ComponentArrays))
    
    # Test initial objective (should match terminal cost version: ~6.062277)
    @test isapprox(optprob.f(optprob.u0, optprob.p), 6.062277, atol = 1.0e-4)
    
    # Test bounds propagation for control u (should have bounds [0, 1])
    # Note: α_int and β_int have Inf bounds, so we skip those
    u_ub = optprob.ub[optprob.lb .== 0.0]  # Lower bound 0 means control variable
    u_lb = optprob.lb[optprob.lb .== 0.0]
    @test all(u_ub .<= 1.0 + 1.0e-6)
    @test all(u_lb .>= 0.0 - 1.0e-6)
end

@testset "MTK Symbolic Interface - Single Shooting IPOPT" begin
    # Full optimization test using symbolic interface with Integral
    # Uses SAME dynamics as main test for comparable numerical results
    
    # Define system without explicit cost state (Integral will add it)
    @variables begin
        x_opt(t) = 0.5, [tunable = false]
        y_opt(t) = 0.7, [tunable = false]
    end
    @variables begin
        u_opt(t) = 0.0, [input = true, bounds = (0.0, 1.0)]
    end
    
    # Same parameters as main test (tunable with same bounds)
    @parameters begin
        α_opt = 1.0, [bounds = (0.0, Inf)]
        β_opt = 1.0, [bounds = (0.0, Inf)]
    end
    
    # Same dynamics as main test (without explicit c state since Integral adds it)
    eqs_opt = [
        D(x_opt) ~ x_opt - β_opt * x_opt * y_opt - 0.4 * u_opt * x_opt,
        D(y_opt) ~ -y_opt + α_opt * x_opt * y_opt - 0.2 * u_opt * y_opt,
    ]
    
    # Lagrangian cost (same as main test's c state)
    lagrangian_opt = Integral(t in (0.0, 12.0))(
        (x_opt - 1.0)^2 + (y_opt - 1.0)^2
    )
    
    @named lotka_opt = ODESystem(eqs_opt, t)
    
    # Create optimization layer
    optlayer = DynamicOptimizationLayer(
        lotka_opt,
        [],
        u_opt => cgrid,  # Use variable symbol, not call
        lagrangian_opt;
        algorithm = Tsit5()
    )
    
    ps, st = LuxCore.setup(rng, optlayer)
    
    # Test type inference
    @test_nowarn @inferred first(optlayer(nothing, ps, st))
    
    # Create and solve optimization problem
    optprob = OptimizationProblem(optlayer, AutoForwardDiff(), vectorizer = Val(:ComponentArrays))
    
    # Test initial objective (should match main test: ~6.062277)
    @test isapprox(optprob.f(optprob.u0, optprob.p), 6.062277, atol = 1.0e-4)
    
    # Solve with IPOPT
    sol = solve(
        optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
        hessian_approximation = "limited-memory"
    )
    
    # Verify successful optimization
    @test SciMLBase.successful_retcode(sol)
    
    # Test final objective (should match main test: ~1.344336)
    @test isapprox(sol.objective, 1.344336, atol = 1.0e-4)
    
    # Verify optimized parameters
    p_opt = sol.u .+ zero(sol.u)
    @test length(p_opt.controls.u) == N
end

@testset "MTK Symbolic Interface - Multiple Shooting with Integral" begin
    # Multiple shooting optimization using symbolic interface
    # Uses SAME dynamics as main test for comparable numerical results
    
    @variables begin
        x_ms(t) = 0.5, [tunable = false]
        y_ms(t) = 0.7, [tunable = false]
    end
    @variables begin
        u_ms(t) = 0.0, [input = true, bounds = (0.0, 1.0)]
    end
    
    # Same parameters as main test (tunable with same bounds)
    @parameters begin
        α_ms = 1.0, [bounds = (0.0, Inf)]
        β_ms = 1.0, [bounds = (0.0, Inf)]
    end
    
    # Same dynamics as main test (without explicit c state since Integral adds it)
    eqs_ms = [
        D(x_ms) ~ x_ms - β_ms * x_ms * y_ms - 0.4 * u_ms * x_ms,
        D(y_ms) ~ -y_ms + α_ms * x_ms * y_ms - 0.2 * u_ms * y_ms,
    ]
    
    # Lagrangian cost (same as main test's c state)
    lagrangian_ms = Integral(t in (0.0, 12.0))(
        (x_ms - 1.0)^2 + (y_ms - 1.0)^2
    )
    
    @named lotka_ms = ODESystem(eqs_ms, t)
    
    # Create optimization layer with multiple shooting
    optlayer = DynamicOptimizationLayer(
        lotka_ms,
        [],
        u_ms => cgrid,  # Use variable symbol, not call
        lagrangian_ms;
        algorithm = Tsit5(),
        shooting = [0.0, 3.0, 6.0, 9.0]  # Multiple shooting points
    )
    
    ps, st = LuxCore.setup(rng, optlayer)
    
    # Test shooting constraints are present (2 states × 3 intervals = 6)
    @test length(optlayer.lcons) == length(optlayer.ucons) == 6
    
    # Test evaluation
    result = @inferred first(optlayer(nothing, ps, st))
    @test result > 0
    
    # Create optimization problem
    optprob = OptimizationProblem(optlayer, AutoForwardDiff(), vectorizer = Val(:ComponentArrays))
    
    # Test initial objective (should match main ms test: ~4.966904)
    @test isapprox(optprob.f(optprob.u0, optprob.p), 4.966904, atol = 1.0e-4)
    
    # Test constraint evaluation (should match main ms test)
    res = zeros(6)
    @inferred first(optlayer(res, ps, st))
    @test isapprox(res, [-1.375755, -0.223574, -1.375755, -0.223574, -1.375755, -0.223574], atol = 1.0e-4)
    
    # Solve with IPOPT
    sol = solve(
        optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
        hessian_approximation = "limited-memory"
    )
    
    # Verify successful optimization
    @test SciMLBase.successful_retcode(sol)
    
    # Test final objective (should match main ms test: ~1.344336)
    @test isapprox(sol.objective, 1.344336, atol = 1.0e-4)
end