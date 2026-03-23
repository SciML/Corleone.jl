using Test
using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using ComponentArrays, ForwardDiff
using Optimization
using OptimizationMOI, Ipopt
using LuxCore, Random

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

@testset "MTK SingleShootingLayer" begin
    layer = SingleShootingLayer(lotka, u(t) => 0.0:0.1:11.9, algorithm = Tsit5(), tspan = (0., 12.0))
    ps, st = LuxCore.setup(rng, layer)
    sol, _ = layer(nothing, ps, st)
    @test length(sol.t) > 1
    @test length(sol.u) == length(sol.t)
end

@testset "MTK DynamicOptimizationLayer (explicit cost/constraints)" begin
    dynopt = DynamicOptimizationLayer(lotka, cost, cons...; controls = [u(t) => 0.0:0.1:11.9], algorithm = Tsit5())
    ps, st = LuxCore.setup(rng, dynopt)
    obj, _ = dynopt(nothing, ps, st)
    @test obj isa Number
    @test isfinite(obj)
end

@testset "MTK DynamicOptimizationLayer (convenience)" begin
    dynopt = DynamicOptimizationLayer(lotka, u(t) => 0.0:0.1:11.9, algorithm = Tsit5())
    ps, st = LuxCore.setup(rng, dynopt)
    obj, _ = dynopt(nothing, ps, st)
    @test obj isa Number
    @test isfinite(obj)
end
