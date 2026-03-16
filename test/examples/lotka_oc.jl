using Corleone
using OrdinaryDiffEqTsit5
using Test
using Random
using LuxCore
using ComponentArrays
using Optimization, OptimizationMOI, Ipopt

using SciMLSensitivity
using SciMLSensitivity.ReverseDiff
using SciMLSensitivity.Zygote
using SymbolicIndexingInterface

rng = Random.default_rng()

function lotka_dynamics!(du, u, p, t)
    du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
    return
end

tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [0.0, 1.0, 1.0]
prob = ODEProblem(
    ODEFunction(lotka_dynamics!, sys = SymbolCache([:x, :y, :c], [:u, :α, :β], :t)),
    u0,
    tspan,
    p0;
    abstol = 1.0e-8,
    reltol = 1.0e-6,
)

cgrid = collect(0.0:0.1:11.9)
N = length(cgrid)
controls = (
    ControlParameter(
        cgrid;
        name = :u,
        bounds = t -> (zero(t), zero(t) .+ 1),
        controls = (rng, t) -> zeros(eltype(t), length(t)),
    ),
    FixedControlParameter(; name = :α, controls = (rng, t) -> [1.0]),
    FixedControlParameter(; name = :β, controls = (rng, t) -> [1.0]),
)

layer = SingleShootingLayer(prob, controls...; algorithm = Tsit5(), quadrature_indices = [3])

ps, st = LuxCore.setup(rng, layer)

sol, _ = layer(nothing, ps, st)

@test sol.t == getsym(sol, :t)(sol)
@test all(sol.p[2] .== getsym(sol, :α)(sol))
@test all(sol.p[3] .== getsym(sol, :β)(sol))
@test ps.controls.u == sol.ps[:u]

x = reduce(hcat, sol.u)

for (i, sym) in enumerate((:x, :y, :c))
    getter = getsym(sol, sym)
    @test getter(sol) == x[i, :]
end

@test_nowarn @inferred first(layer(nothing, ps, st))

@test allunique(sol.t)
@test LuxCore.parameterlength(layer) == N + 2


for AD in (AutoForwardDiff(), AutoReverseDiff(), AutoZygote())
    prob = ODEProblem(
        ODEFunction(lotka_dynamics!, sys = SymbolCache([:x, :y, :c], [:u, :α, :β], :t)),
        u0,
        tspan,
        p0;
        abstol = 1.0e-8,
        reltol = 1.0e-6,
        sensealg = AD == AutoZygote() ? ForwardDiffSensitivity() : SciMLBase.NoAD(),
    )

    layer = SingleShootingLayer(prob, controls...; algorithm = Tsit5(), quadrature_indices = [3])

    ps, st = LuxCore.setup(rng, layer)

    p = ComponentArray(ps)

    loss = let layer = layer, ax = getaxes(p)
        (p, st) -> begin
            traj, _ = layer(nothing, ComponentArray(p, ax), st)
            traj[:c][end]
        end
    end
    #@test size(p, 1) == LuxCore.parameterlength(layer)
    #optprob = OptimizationProblem(layer, AD, Val(:ComponentArrays), loss = :c)
    optfun = OptimizationFunction(loss, AD)
    optprob = OptimizationProblem(
        optfun, collect(p), st;
        lb = zero(p), ub = zero(p) .+ 1
    )

    @test isapprox(optprob.f(optprob.u0, optprob.p), 6.062277454291031, atol = 1.0e-4)

    sol = solve(
        optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
        hessian_approximation = "limited-memory"
    )

    @test SciMLBase.successful_retcode(sol)
    @test isapprox(sol.objective, 1.344336, atol = 1.0e-4)

    p_opt = sol.u .+ zero(p)

    @test isempty(p_opt.initial_conditions)
    @test length(p_opt.controls.u) == N
end
