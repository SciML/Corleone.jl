using Pkg
Pkg.activate(@__DIR__)
using Corleone
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using LuxCore
using Random

using CairoMakie
using Optimization
using OptimizationMOI
using Ipopt
using blockSQP

function batch_reactor(u, p, t)
    x,y  = u
    return [ -4.e3 * exp(-2500 / p[1]) * x^2
            4.e3 * exp(-2500 / p[1]) * x^2 - 62.e4 * exp(-5000 / p[1]) * y^2
            ]
end

tspan = (0., 1.0)
u0 = [1.0, 0.0]
p = [300.0]

prob =  ODEProblem(batch_reactor, u0, tspan, p)

N = 20
control = ControlParameter(
    collect(LinRange(tspan..., N+1))[1:end-1], name = :u, controls = 300*ones(N), bounds = (298.0,398.0)
)
layer = Corleone.SingleShootingLayer(prob, Tsit5(), controls = (1 => control,))

ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer) .|> ComponentArray

loss = let layer = layer, st = st, ax = getaxes(p)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        -last(sols.u)[2]
    end
end

loss(collect(p), nothing)

optfun = OptimizationFunction(
    loss, AutoForwardDiff(),
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub)
)
uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300
)


optsol, _ = layer(nothing, uopt + zero(p), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
scatterlines!(ax, optsol, idxs=[1,2])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2,1])
stairs!(ax1, optsol, vars=[:u₁])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
f