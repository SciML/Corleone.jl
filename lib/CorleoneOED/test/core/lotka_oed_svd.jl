using Corleone: SciMLBase
using Corleone
using CorleoneOED
using OrdinaryDiffEqTsit5
using ComponentArrays
using LuxCore
using StableRNGs

using Test
using Optimization
using OptimizationMOI
using Ipopt
using LinearAlgebra

function lotka_dynamics(u, p, t)
    return [
        u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1];
        -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    ]
end

tspan = (0.0, 12.0)
u0 = [0.5, 0.7]
p0 = [0.0, 1.0, 1.0]

prob = ODEProblem{false}(
    lotka_dynamics, u0, tspan, p0,
)

cgrid = 0.0:0.25:11.75
control = ControlParameter(
    collect(cgrid), name = :fishing, bounds = (0.0, 1.0)
)

layer = SingleShootingLayer(prob, Tsit5(), controls = (1 => control,), bounds_p = ([1.0, 1.0], [1.0, 1.0]))

tgrid1 = 0.0:0.5:11.5
tgrid2 = 0.0:0.15:11.75

oed = OEDLayer{false}(
    layer,
    params = [2, 3],
    measurements = [
        ControlParameter(collect(tgrid1), controls = ones(length(tgrid1)), bounds = (0.0, 1.0)),
        ControlParameter(collect(tgrid2), controls = ones(length(tgrid2)), bounds = (0.0, 1.0)),
    ],
    observed = (u, p, t) -> u[1:2],
    svd = true
)

ps, st = LuxCore.setup(StableRNG(1), oed)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(oed) .|> ComponentArray

sol, _ = oed(nothing, p, st)

@test sol.u[1] == vcat(u0, zeros(4 + 3 + 1), ones(2))
@test size.(sensitivities(oed, sol))[1] == (2, 2) # 2 singular values to describe F
@test collect(lb) == vcat(ones(2), zeros(length(cgrid) + length(tgrid1) + length(tgrid2)))
@test collect(ub) == vcat(ones(2), ones(length(cgrid) + length(tgrid1) + length(tgrid2)))

res = zeros(3)
@test_nowarn CorleoneOED.get_sampling_sums!(res, oed, nothing, ps, st)
@test [12.0, 12.0, 0.0] == res
@test CorleoneOED.get_sampling_sums(oed, nothing, ps, st) == res[1:2]
@test_nowarn @inferred CorleoneOED.get_sampling_sums(oed, nothing, ps, st)


optprob = OptimizationProblem(oed, ACriterion(), M = [4.0, 4.0])

@test optprob.f(optprob.u0, optprob.p) ≈ 0.05352869250783344

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 100,
    print_level = 0,
)

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory", # sb="yes",
    max_iter = 100,
    print_level = 0,
)

@testset "Solution" begin
    @test uopt.objective ≈ 0.0370750919
    @test uopt.retcode == SciMLBase.ReturnCode.Success
    opt_p = zero(p) .+ uopt
    sol, _ = oed(nothing, opt_p, st)
    c1 = reduce(vcat, map(Base.Fix2(getindex, 10), sol.u))
    u1 = reduce(vcat, map(Base.Fix2(getindex, 11), sol.u))
    u2 = reduce(vcat, map(Base.Fix2(getindex, 12), sol.u))
    @test sol.u[1][1:2] == u0
    @test sensitivities(oed, sol)[end] ≈ [-1.2238537938389058 -0.17983968932011046; -2.1982709801704483 -4.869232906716779]
    F, _ = fisher_information(oed, nothing, opt_p, st)
    @test F ≈ [88.89549299146556 13.63402689709759; 13.63402689709759 41.72266803144628]
    @test sol.t[c1 .> 0.1] == [0.0, 0.15, 0.25, 0.3, 0.45, 0.5, 0.6, 0.75, 0.9, 1.0, 4.8, 4.95, 5.0, 5.1, 5.25, 5.4, 5.5, 5.55, 5.7, 5.75, 5.85, 6.0, 6.15, 6.25, 6.3, 6.45, 6.5, 6.6, 6.75, 6.9, 7.0, 7.05, 7.2, 7.25, 7.35, 7.5, 7.65, 7.75]
    @test sol.t[u1 .> 0.1] == [2.55, 2.7, 2.75, 2.85, 3.0, 3.15, 3.25, 3.3, 3.45, 3.5, 3.6, 3.75, 3.9, 4.0, 4.05, 4.2, 4.25, 4.35, 4.5, 4.65, 4.75, 4.8, 4.95, 5.0, 10.65, 10.75, 10.8, 10.95, 11.0, 11.1, 11.25, 11.4, 11.5, 11.55, 11.7, 11.75, 12.0]
    @test sol.t[u2 .> 0.1] == [3.0, 3.15, 3.25, 3.3, 3.45, 3.5, 3.6, 3.75, 3.9, 4.0, 4.05, 4.2, 4.25, 4.35, 4.5, 4.65, 4.75, 4.8, 4.95, 5.0, 5.1, 5.25, 5.4, 10.65, 10.75, 10.8, 10.95, 11.0, 11.1, 11.25, 11.4, 11.5, 11.55, 11.7, 11.75, 12.0]
end
