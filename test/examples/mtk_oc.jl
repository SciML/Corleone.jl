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
using SciMLSensitivity: ForwardDiffSensitivity

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

# Note: We don't use @parameters here because MTK parameters are tunable by default,
# and MTK's generated function wrappers don't work with ForwardDiff Dual types through
# tunable parameters. The α=1.0 and β=1.0 values are hardcoded in the equations below.

# Lotka-Volterra dynamics with control
# dx/dt = x - β*x*y - 0.4*u*x   with β=1.0 hardcoded
# dy/dt = -y + α*x*y - 0.2*u*y  with α=1.0 hardcoded  
# Cost: (x - 1)^2 + (y - 1)^2 integrated over time
eqs = [
    D(x) ~ x - 1.0 * x * y - 0.4 * u * x,
    D(y) ~ -y + 1.0 * x * y - 0.2 * u * y,
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
        algorithm=Tsit5(),
        tspan=(0.0, 12.0),
        quadrature_indices=[3]  # Index of cost variable
    )

    ps, st = LuxCore.setup(rng, layer)
    sol, _ = layer(nothing, ps, st)

    # Time access
    @test sol.t == getsym(sol, :t)(sol)

    # Control values
    @test ps.controls.u == sol.ps[:u][1:(end-1)]

    # State values via getsym - check lengths match
    @test length(getsym(sol, x)(sol)) == length(sol.t)
    @test length(getsym(sol, y)(sol)) == length(sol.t)
    @test length(getsym(sol, c)(sol)) == length(sol.t)

    # Type inference
    @test_nowarn @inferred first(layer(nothing, ps, st))
    @test allunique(sol.t)

    # Parameter length: N control values for u (no tunable parameters)
    @test LuxCore.parameterlength(layer) == N

    # Test bounds - u bounds: [0, 1]
    lb = get_lower_bound(layer)
    ub = get_upper_bound(layer)

    @test all(lb.controls.u .>= 0.0 - 1.0e-6)
    @test all(ub.controls.u .<= 1.0 + 1.0e-6)

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
        algorithm=Tsit5(),
        tspan=(0.0, 1.0)
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

    # No tunable parameters (α and β hardcoded in equations)
    @test length(keys(lb.controls)) == 1
    @test :u in keys(lb.controls)
end

@testset "MTK Parameter Access Patterns" begin
    layer = SingleShootingLayer(
        lotka_system,
        [],
        u => 0.0:0.1:1.0;
        algorithm=Tsit5(),
        tspan=(0.0, 1.0)
    )

    ps, st = LuxCore.setup(rng, layer)
    traj, _ = layer(nothing, ps, st)

    # Test getp for control parameters
    u_getter = getp(traj, u)

    @test u_getter(traj) isa Vector

    # Test sizes
    @test length(u_getter(traj)) == length(traj.t)
end

@testset "MTK DynamicOptimizationLayer" begin
    # Test basic DynamicOptimizationLayer construction and evaluation
    layer = SingleShootingLayer(
        lotka_system,
        [],
        u => cgrid;
        algorithm=Tsit5(),
        tspan=(0.0, 1.0),
        quadrature_indices=[3]
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
        algorithm=Tsit5(),
        tspan=(0.0, 1.0),
        quadrature_indices=[3]  # c is the cost variable
    )

    ps, st = LuxCore.setup(rng, layer)
    traj, _ = layer(nothing, ps, st)

    # c should accumulate the cost over time
    c_vals = getsym(traj, c)(traj)

    # Cost should be non-negative and increase over time
    @test all(c_vals .>= -1.0e-6)
    @test c_vals[end] >= c_vals[1]  # Cost integral should grow
end

eqs2 = [
    D(x) ~ x - x * y - 0.4 * u * x,
    D(y) ~ -y + x * y - 0.2 * u * y,
    D(c) ~ (x - 1.0)^2 + (y - 1.0)^2,
]

@named lotka_system2 = ODESystem(eqs2, t)

@testset "MTK Single Shooting IPOPT Optimization" begin
    # NOTE: This test is broken due to a fundamental MTK limitation.
    # MTK's RuntimeGeneratedFunction uses FunctionWrappersWrappers.jl which doesn't
    # support ForwardDiff.Dual types. The error "No matching function wrapper was found!"
    # occurs when ForwardDiff tries to differentiate through MTK's generated ODE functions.
    #
    # This affects BOTH NoAD() and ForwardDiffSensitivity() sensealg configurations.
    #
    # WORKAROUND: Use plain Julia ODEProblem (not MTK) for ForwardDiff optimization.
    # The lotka_oc.jl tests demonstrate working optimization with plain Julia functions.
    # See: https://github.com/SciML/ModelingToolkit.jl/issues regarding ForwardDiff support.
    
    # The tests below would run if MTK supported ForwardDiff Dual types:
    
    # The following tests would run if MTK supported ForwardDiff:
    layer = SingleShootingLayer(
        lotka_system2,
        [],  # No initial condition overrides
        u => cgrid;
        algorithm=Tsit5(),
        tspan=(0.0, 12.0),
        quadrature_indices=[3]  # Index of cost variable
    )

    ps, st = LuxCore.setup(rng, layer)
    sol, _ = layer(nothing, ps, st)

    # Verify basic layer functionality first
    @test sol.t == getsym(sol, :t)(sol)

    # Test bounds before optimization
    lb = get_lower_bound(layer)
    ub = get_upper_bound(layer)
    @test all(lb.controls.u .>= 0.0 - 1.0e-6)
    @test all(ub.controls.u .<= 1.0 + 1.0e-6)

    # Run optimization with AutoForwardDiff (MTK supports only ForwardDiff)
    # Must use sensealg=NoAD() to avoid ForwardDiff function wrapper issues with MTK
    layer = remake(layer, sensealg=SciMLBase.NoAD())
    optlayer = DynamicOptimizationLayer(layer, :(c(12.0)))
    ps, st = LuxCore.setup(rng, optlayer)

    # Test type inference
    @test_nowarn @inferred first(optlayer(nothing, ps, st))

    # Create optimization problem
    optprob = OptimizationProblem(optlayer, AutoForwardDiff(), vectorizer=Val(:ComponentArrays))
    p = ComponentArray(ps)

    # Test initial objective value (should match lotka_oc.jl: ~6.062277)
    @test isapprox(optprob.f(optprob.u0, optprob.p), 6.062277454291031, atol=1.0e-4)

    # Test bounds
    @test all(optprob.ub .== 1.0)
    @test all(optprob.lb .== 0.0)

    # Solve with IPOPT
    sol = solve(
        optprob, Ipopt.Optimizer(), max_iter=1000, tol=5.0e-6,
        hessian_approximation="limited-memory"
    )

    # Verify successful optimization
    @test SciMLBase.successful_retcode(sol)

    # Test final objective (should match lotka_oc.jl: ~1.344336)
    @test isapprox(sol.objective, 1.344336, atol=1.0e-4)

    # Verify optimized parameters
    p_opt = sol.u .+ zero(p)
    @test isempty(p_opt.initial_conditions)
    @test length(p_opt.controls.u) == N
end

@testset "MTK Multiple Shooting IPOPT Optimization" begin
    # Multiple shooting optimization test using ForwardDiffSensitivity
    layer = SingleShootingLayer(
        lotka_system2,
        [],
        u => cgrid;
        bounds_ic=(t0) -> (zeros(3), fill(Inf, 3)),
        algorithm=Tsit5(),
        tspan=(0.0, 12.0),
        quadrature_indices=[c],
        sensealg=ForwardDiffSensitivity()
    )

    ms_layer = MultipleShootingLayer(layer, 0.0, 3.0, 6.0, 9.0)
    ps, st = LuxCore.setup(rng, ms_layer)
    traj, _ = ms_layer(nothing, ps, st)

    # Test shooting constraints count
    @test Corleone.get_number_of_shooting_constraints(ms_layer) == 6

    # Run optimization with AutoForwardDiff
    ms_layer = remake(ms_layer, sensealg=ForwardDiffSensitivity())
    optlayer = DynamicOptimizationLayer(ms_layer, :(c(12.0)))

    # Test constraint bounds
    @test length(optlayer.lcons) == length(optlayer.ucons) == 6
    @test optlayer.lcons == optlayer.ucons

    ps, st = LuxCore.setup(rng, optlayer)

    # Test type inference
    objectiveval = @inferred first(optlayer(nothing, ps, st))

    # Test initial objective (should match lotka_oc.jl: ~4.966904)
    @test isapprox(objectiveval, 4.9669040432037574, atol=1.0e-4)

    # Test constraint evaluation
    res = zeros(6)
    @inferred first(optlayer(res, ps, st))
    @test isapprox(res, [-0.2235735751355118,-1.3757549609694821,-0.2235735751355118,-1.3757549609694821,-0.2235735751355118,-1.3757549609694821], atol=1.0e-4)

    # Create and solve optimization problem
    optprob = OptimizationProblem(optlayer, AutoForwardDiff(), vectorizer=Val(:ComponentArrays))
    sol = solve(
        optprob, Ipopt.Optimizer(), max_iter=1000, tol=5.0e-6,
        hessian_approximation="limited-memory"
    )

    # Verify successful optimization
    @test SciMLBase.successful_retcode(sol)

    # Test final objective (should match lotka_oc.jl: ~1.344336)
    @test isapprox(sol.objective, 1.344336, atol=1.0e-4)
end

@testset "MTK Symbolic Interface - Lagrangian Cost with Integral" begin
    # Test using Symbolics.Integral for Lagrangian cost term
    # ∫₀ᴰ ((x-1)² + (y-1)²) dt
    
    @variables begin
        x_int(t) = 0.5, [tunable = false]
        y_int(t) = 0.7, [tunable = false]
    end
	@constants begin 
		c[1:2] = [0.4, 0.2]
	end
    @variables begin
        u_int(t) = 0.0, [input = true, bounds = (0.0, 1.0)]
    end

    # NOTE: No @parameters to avoid MTK + ForwardDiff incompatibility
    # α=1, β=1 hardcoded in equations
    eqs_int = [
		D(x_int) ~ x_int - 1.0 * x_int * y_int - c[1] * u_int * x_int,
		D(y_int) ~ -y_int + 1.0 * x_int * y_int - c[2] * u_int * y_int,
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
        lagrangian, 
		EvalAt(12.0)(x_int) ~ 1.0, 
		EvalAt(12.0)(y_int) ~ 1.0, 
		(EvalAt(12.0)(x_int)^2 +EvalAt(12.0)(y_int)^2) >= c[1] + c[2] 
		;  # Pass Integral expression directly
        algorithm=Tsit5()
    )

    ps, st = LuxCore.setup(rng, optlayer)

    # Test evaluation
    result = @inferred first(optlayer(nothing, ps, st))
    
    @test length(optlayer.lcons) == length(optlayer.ucons) == 3
    @test optlayer.lcons != optlayer.ucons

	# Create optimization problem
    optprob = OptimizationProblem(optlayer, AutoForwardDiff(), vectorizer=Val(:ComponentArrays))
    p = ComponentArray(ps)

    # Test initial objective value (should match lotka_oc.jl: ~6.062277)
    @test isapprox(optprob.f(optprob.u0, optprob.p), 6.062277454291031, atol=1.0e-4)
	res = zeros(3) 
	
	@test isapprox([-0.5262052216721573, 0.2607650855766033, -1.2140100929797093], optprob.f.cons(res, optprob.u0, optprob.p), atol = 1e-4)
    
	# Test bounds
    @test all(optprob.ub .== 1.0)
    @test all(optprob.lb .== 0.0)

    # Solve with IPOPT
    sol = solve(
        optprob, Ipopt.Optimizer(), max_iter=1000, tol=5.0e-6,
        hessian_approximation="limited-memory"
    )

    # Verify successful optimization
    @test SciMLBase.successful_retcode(sol)

    # Test final objective (should match lotka_oc.jl: ~1.344336)
    @test isapprox(sol.objective, 1.344336, atol=1.0e-4)

    # Verify optimized parameters
    p_opt = sol.u .+ zero(p)
    @test isempty(p_opt.initial_conditions)
end

