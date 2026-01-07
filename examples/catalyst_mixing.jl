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

function catalyst_mixing(du, u, p, t)
    x,y  = u
    du[1] = p[1] * (10 * y - x)
    du[2] =  p[1] * (x - 10 * y) - (1 - p[1]) * y
end

tspan = (0., 1.0)
u0 = [1.0, 0.0]
p = [300.0]

prob = ODEProblem(catalyst_mixing, u0, tspan, p)

N = 20
control = ControlParameter(
    collect(LinRange(tspan..., N+1))[1:end-1], name = :u, controls = zeros(N) .+ 0.5, bounds = (0.,1.)
)
layer = Corleone.SingleShootingLayer(prob, Tsit5(), controls=(1 => control,))
ps, st = LuxCore.setup(Random.default_rng(), layer)

optprob = OptimizationProblem(
    layer, :(-1 + x₁ + x₂)
)
uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300
)

optsol, _ = layer(nothing, uopt + zero(ComponentArray(ps)), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
scatterlines!(ax, optsol, vars=[:x₁, :x₂])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2,1])
stairs!(ax1, optsol, vars=[:u₁])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
f