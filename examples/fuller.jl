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


function fuller(du, u, p, t)
    du[1] = u[2]
    du[2] = 1.0 - 2.0 * p[1]
    return du[3] = u[1]^2
end

tspan = (0.0, 1.0)
u0 = [1.0e-2, 0.0, 0.0]
p = [0.5]

prob = ODEProblem(
    fuller, u0, tspan, p,
    abstol = 1.0e-8, reltol = 1.0e-8
)
plot(solve(prob, Tsit5()))

dt = 0.01
cgrid = collect(0.0:dt:1.0)[1:(end - 1)]
control = ControlParameter(
    cgrid, name = :u, controls = rand(length(cgrid)), bounds = (0, 1)
)

layer = SingleShootingLayer(prob, Tsit5(), controls = (1 => control,))

ps, st = LuxCore.setup(Random.default_rng(), layer)

constraints = Dict(
    :x₁ => (t = last(tspan), bounds = (1.0e-2, 1.0e-2)),
    :x₂ => (t = last(tspan), bounds = (0.0, 0.0))
)

optprob = OptimizationProblem(
    layer, :x₃, constraints = constraints
)


uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-12,
    hessian_approximation = "limited-memory",
    max_iter = 250
)

optsol, _ = layer(nothing, uopt + zero(ComponentArray(ps)), st)

f = Figure()
ax = CairoMakie.Axis(f[1, 1])
scatterlines!(ax, optsol, vars = [:x₁, :x₂, :x₃])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2, 1])
stairs!(ax1, optsol, vars = [:u₁])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
display(f)
