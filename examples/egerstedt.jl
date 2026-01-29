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

function egerstedt(du, u, p, t)
    x, y, _ = u
    u1, u2, u3 = p
    du[1] = -x * u1 + (x + y) * u2 + (x - y) * u3
    du[2] = (x + 2 * y) * u1 + (x - 2 * y) * u2 + (x + y) * u3
    return du[3] = x^2 + y^2
end

tspan = (0.0, 1.0)
u0 = [0.5, 0.5, 0.0]
p = 1 / 3 * ones(3)

prob = ODEProblem(egerstedt, u0, tspan, p)

N = 20
cgrid = collect(LinRange(tspan..., N + 1))[1:(end - 1)]
c1 = ControlParameter(
    cgrid, name = :u1, bounds = (0.0, 1.0), controls = LinRange(0.0, 0.2, N)
)
c2 = ControlParameter(
    cgrid, name = :u2, bounds = (0.0, 1.0), controls = LinRange(0.3, 0.5, N)
)
c3 = ControlParameter(
    cgrid, name = :u3, bounds = (0.0, 1.0), controls = LinRange(0.6, 0.8, N)
)

layer = Corleone.SingleShootingLayer(prob, Tsit5(), controls = ([1, 2, 3] .=> [c1, c2, c3]))
ps, st = LuxCore.setup(Random.default_rng(), layer)

# Define constraints with expressions and corresponding timepoints to evaluate at
cons = :(u₁ + u₂ + u₃)
timepoints = vcat(cgrid[2:end], last(tspan))

constraints = Dict(cons => (t = timepoints, bounds = (ones(length(timepoints)), ones(length(timepoints)))))

optprob = OptimizationProblem(layer, :x₃; constraints = constraints)

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300
)

optsol, _ = layer(nothing, uopt + zero(ComponentArray(ps)), st)

f = Figure()
ax = CairoMakie.Axis(f[1, 1])
scatterlines!(ax, optsol, vars = [:x₁, :x₂, :x₃])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2, 1])
stairs!(ax1, optsol, vars = [:u₁, :u₂, :u₃])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
f

### Multiple Shooting
shooting_points = [0.0, 0.25, 0.75]
layer = Corleone.MultipleShootingLayer(prob, Tsit5(), shooting_points...; controls = ([1, 2, 3] .=> [c1, c2, c3]))
ps, st = LuxCore.setup(Random.default_rng(), layer)

cons = :(u₁ + u₂ + u₃)
timepoints = vcat(cgrid[2:end], last(tspan))

constraints = Dict(cons => (t = timepoints, bounds = (ones(length(timepoints)), ones(length(timepoints)))))

optprob = OptimizationProblem(layer, :x₃; constraints = constraints)

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300
)

blocks = Corleone.get_block_structure(layer)

uopt = solve(
    optprob, BlockSQPOpt(),
    opttol = 1.0e-6,
    options = blockSQP.sparse_options(),
    sparsity = blocks,
    maxiters = 300 # 165
)

optsol, _ = layer(nothing, uopt + zero(ComponentArray(ps)), st)

f = Figure()
ax = CairoMakie.Axis(f[1, 1])
scatterlines!(ax, optsol, vars = [:x₁, :x₂, :x₃])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2, 1])
stairs!(ax1, optsol, vars = [:u₁, :u₂, :u₃])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
f
