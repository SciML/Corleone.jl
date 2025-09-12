using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))


using CorleoneCore
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using LuxCore
using Random

using CairoMakie
using BenchmarkTools
using Zygote
using ForwardDiff

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
    collect(LinRange(tspan..., N+1))[1:end-1], name = :u, controls = 300*ones(N)
)
layer = CorleoneCore.SingleShootingLayer(prob, Tsit5(),Int64[],[1], (control,))

ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)


loss = let layer = layer, st = st, ax = getaxes(p)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        -sols[:x₂][end]
    end
end

loss(collect(p), nothing)


optfun = OptimizationFunction(
    loss, AutoForwardDiff(),
)

lb, ub = copy(p), copy(p)
lb.controls .= 298.0
ub.controls .= 398.0


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