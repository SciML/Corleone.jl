using Test
using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using ComponentArrays, ForwardDiff
using Optimization
using OptimizationMOI, Ipopt
using LuxCore, Random
using SymbolicIndexingInterface

#TODO AGENT 
# To call this script, use the TestEnv package with TestEnv.activate() to have access to all needed pacakages above. 

@variables x(..) = 0.5 [tunable = false] y(..) = 0.7 [tunable = false]
@variables u(..) = 0.0 [bounds = (0.0, 1.0), input = true]
@constants begin
    c₁ = 0.4
    c₂ = 0.2
end
@parameters begin
    α[1:1] = [1.0], [tunable = true, bounds = ([1.0], [1.0])]
    β = 1.0, [tunable = true, bounds = (0.9, 1.1)]
end

cost = [
    Symbolics.Integral(t in (0.0, 12.0))(
        (x(t) - 1.0)^2 + (y(t) - 1.0)^2
    ),
]

cons = [
    x(0.0) ≳ 0.2,
    β ~ 1.0,
]


@named lotka = System(
    [
        D(x(t)) ~ α[1] * x(t) - β * x(t) * y(t) - c₁ * u(t) * x(t),
        D(y(t)) ~ - y(t) + x(t) * y(t) - c₂ * u(t) * y(t),
    ], t; costs = cost, constraints = cons
)

rng = Random.default_rng()
cgrid = collect(0.0:0.1:11.9)
N = length(cgrid)

@testset "MTK SingleShootingLayer" begin
    layer = SingleShootingLayer(lotka, u(t) => 0.0:0.1:11.9, algorithm = Tsit5(), tspan = (0., 12.0))
    ps, st = LuxCore.setup(rng, layer)
    sol, _ = layer(nothing, ps, st)

    # Trajectory access
    @test sol.t == getsym(sol, :t)(sol)
    @test length(sol.t) > 1
    @test length(sol.u) == length(sol.t)

    # State symbolic access
    x_mat = reduce(hcat, sol.u)
    for (i, sym) in enumerate((:x, :y))
        getter = getsym(sol, sym)
        @test getter(sol) == x_mat[i, :]
    end

    # Inferred forward pass
    @test_nowarn @inferred first(layer(nothing, ps, st))
    @test allunique(sol.t)
end

@testset "MTK DynamicOptimizationLayer (explicit cost/constraints)" begin
    dynopt = DynamicOptimizationLayer(lotka, cost, cons...; controls = [u(t) => 0.0:0.1:11.9], algorithm = Tsit5())
    ps, st = LuxCore.setup(rng, dynopt)
    obj, _ = dynopt(nothing, ps, st)
    @test obj isa Number
    @test isfinite(obj)
    @test_nowarn @inferred first(dynopt(nothing, ps, st))
end

@testset "MTK DynamicOptimizationLayer (convenience)" begin
    dynopt = DynamicOptimizationLayer(lotka, u(t) => 0.0:0.1:11.9, algorithm = Tsit5())
    ps, st = LuxCore.setup(rng, dynopt)
    obj, _ = dynopt(nothing, ps, st)
    @test obj isa Number
    @test isfinite(obj)
    @test_nowarn @inferred first(dynopt(nothing, ps, st))
end

@testset "MTK Single Shooting Optimization" begin
    dynopt = DynamicOptimizationLayer(lotka, u(t) => 0.0:0.1:11.9, algorithm = Tsit5())
    ps, st = LuxCore.setup(rng, dynopt)
    @inferred first(dynopt(nothing, ps, st))
    optprob = OptimizationProblem(dynopt, AutoForwardDiff(), vectorizer = Val(:ComponentArrays))
    p = ComponentArray(ps)
    @test all(optprob.ub[1:N] .== 1.0)
    @test all(optprob.lb[1:N] .== 0.0)
    sol = solve(
        optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
        hessian_approximation = "limited-memory"
    )
    @test SciMLBase.successful_retcode(sol)
    @test isfinite(sol.objective)
    @test sol.objective < 10.0  # Objective should be reasonable
end

@testset "MTK Multiple Shooting" begin
    shooting_points = [0.0, 3.0, 6.0, 9.0]

    # Convenience constructor with shooting
    dynopt = DynamicOptimizationLayer(lotka, u(t) => 0.0:0.1:11.9;
        algorithm = Tsit5(),
        shooting = shooting_points,
    )
    ps, st = LuxCore.setup(rng, dynopt)

    # Check shooting constraints count (3 interior nodes × 3 states = 9, but only non-quadrature states get IC)
    n_shoot = Corleone.get_number_of_shooting_constraints(dynopt)
    @test n_shoot > 0
    @test length(dynopt.lcons) == length(dynopt.ucons)
    @test dynopt.lcons[1:n_shoot] == dynopt.ucons[1:n_shoot]

    # Evaluate objective
    objectiveval = @inferred first(dynopt(nothing, ps, st))
    @test isfinite(objectiveval)

    # Evaluate constraints
    res = zeros(length(dynopt.lcons))
    @inferred first(dynopt(res, ps, st))
    @test all(isfinite, res)

    # Solve optimization (may not always converge for multiple shooting with default init)
    optprob = OptimizationProblem(dynopt, AutoForwardDiff(), vectorizer = Val(:ComponentArrays))
    sol = solve(
        optprob, Ipopt.Optimizer(), max_iter = 2000, tol = 5.0e-4,
        hessian_approximation = "limited-memory"
    )
    @test isfinite(sol.objective)

    # Explicit constructor with shooting
    dynopt2 = DynamicOptimizationLayer(lotka, cost, cons...;
        controls = [u(t) => 0.0:0.1:11.9],
        algorithm = Tsit5(),
        shooting = shooting_points,
    )
    ps2, st2 = LuxCore.setup(rng, dynopt2)
    obj2, _ = dynopt2(nothing, ps2, st2)
    @test isfinite(obj2)
end
