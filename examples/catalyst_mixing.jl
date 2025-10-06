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

function catalyst_mixing(u, p, t)
    x,y  = u
    return [ p[1] * (10 * y - x)
            p[1] * (x - 10 * y) - (1 - p[1]) * y
            ]
end

tspan = (0., 1.0)
u0 = [1.0, 0.0]
p = [300.0]

prob = ODEProblem(catalyst_mixing, u0, tspan, p)

N = 20
control = ControlParameter(
    collect(LinRange(tspan..., N+1))[1:end-1], name = :u, controls = zeros(N) .+ 0.5, bounds = (0.,1.)
)
layer = Corleone.SingleShootingLayer(prob, Tsit5(),[1], (control,))

ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer)

loss = let layer = layer, st = st, ax = getaxes(p)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        -1 + sols[:x₁][end] + sols[:x₂][end]
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
[lines!(ax, optsol.t, optsol[x], label = string(x)) for x in [:x₁, :x₂]]
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2,1])
stairs!(ax1, optsol.t, optsol[:u₁], label = "u₁")
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
f