# Test for MTK integration - similar structure to lotka_oc.jl but MTK-specific
# Note: MTK only supports ForwardDiff for automatic differentiation

using Test
using Corleone
using Corleone: get_lower_bound, get_upper_bound
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using Random
using LuxCore
using SymbolicIndexingInterface

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

# Time grid for control discretization
cgrid = collect(0.0:0.1:11.9)
N = length(cgrid)

@testset "MTK SingleShootingLayer" begin
    # Create SingleShootingLayer with MTK system
    layer = SingleShootingLayer(
        lotka_system,
        [],  # No initial condition overrides
        u => cgrid;  # Control specification
        algorithm = Tsit5(),
        tspan = (0.0, 12.0),
        quadrature_indices = [3]  # Index of cost variable
    )

    ps, st = LuxCore.setup(rng, layer)
    traj, st_new = layer(nothing, ps, st)

    # Test basic properties
    @test length(traj.t) > 0
    @test traj.t == getsym(traj, :t)(traj)

    # Test symbolic indexing with MTK symbols
    # State variables
    x_getter = getsym(traj, x)
    y_getter = getsym(traj, y)
    c_getter = getsym(traj, c)

    @test length(x_getter(traj)) == length(traj.t)
    @test length(y_getter(traj)) == length(traj.t)
    @test length(c_getter(traj)) == length(traj.t)

    # Control parameter via ps interface
    u_vals = traj.ps[u]  # MTK symbolic access
    @test length(u_vals) == length(traj.t)

    # Control parameter via symbol
    u_vals_sym = traj.ps[:u]  # Symbol access
    @test u_vals == u_vals_sym

    # Parameters become ControlParameters - check they exist with correct names
    @test :α in Corleone._control_names(traj)
    @test :β in Corleone._control_names(traj)
    @test :u in Corleone._control_names(traj)

    # Test parameter length
    # N control values for u + 2 fixed parameters (α, β)
    @test LuxCore.parameterlength(layer) == N + 2

    # Test is_observed and is_parameter
    @test SymbolicIndexingInterface.is_observed(traj, u) == true
    @test SymbolicIndexingInterface.is_observed(traj, α) == true
    @test SymbolicIndexingInterface.is_observed(traj, β) == true
    @test SymbolicIndexingInterface.is_parameter(traj, u) == false
    @test SymbolicIndexingInterface.is_parameter(traj, α) == false
    @test SymbolicIndexingInterface.is_parameter(traj, x) == false  # x is a state

    # Test type inference
    @test_nowarn @inferred first(layer(nothing, ps, st))

    # Test that trajectory values are reasonable (populations non-negative with tolerance)
    x_vals = x_getter(traj)
    y_vals = y_getter(traj)
    @test all(x_vals .>= -1.0e-6)
    @test all(y_vals .>= -1.0e-6)
end

@testset "MTK Symbolic Indexing Comprehensive" begin
    layer = SingleShootingLayer(
        lotka_system,
        [],
        u => cgrid;
        algorithm = Tsit5(),
        tspan = (0.0, 1.0)
    )

    ps, st = LuxCore.setup(rng, layer)
    traj, _ = layer(nothing, ps, st)

    # Test all symbolic access patterns work

    # 1. State variables via getsym with MTK symbols
    @test getsym(traj, x)(traj) isa Vector
    @test getsym(traj, y)(traj) isa Vector
    @test getsym(traj, c)(traj) isa Vector

    # 2. Control parameters via ps with MTK symbols
    @test traj.ps[u] isa Vector
    @test traj.ps[α] isa Vector
    @test traj.ps[β] isa Vector

    # 3. Same via plain Symbols
    @test traj.ps[:u] == traj.ps[u]
    @test traj.ps[:α] == traj.ps[α]
    @test traj.ps[:β] == traj.ps[β]

    # 4. Time access
    @test getsym(traj, :t)(traj) == traj.t

    # 5. Sizes match
    n_timepoints = length(traj.t)
    @test length(traj.ps[u]) == n_timepoints
    @test length(getsym(traj, x)(traj)) == n_timepoints
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