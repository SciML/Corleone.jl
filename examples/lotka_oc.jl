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
# using SymbolicIndexingInterface

function lotka_dynamics(du, u, p, t)
    du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    return du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
end

tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [0.0, 1.0, 1.0]

prob = ODEProblem(lotka_dynamics, u0, tspan, p0) #, sensealg=SciMLBase.NoAD())

cgrid = collect(0.0:0.1:11.9)
control = ControlParameter(
    cgrid, name = :fishing, bounds = (0.0, 1.0), controls = ones(length(cgrid))
)

# Single Shooting
layer = Corleone.SingleShootingLayer(prob, Tsit5(), controls = (1 => control,), bounds_p = ([1.0, 1.0], [1.0, 1.0]))
ps, st = LuxCore.setup(Random.default_rng(), layer)

optprob = OptimizationProblem(layer, :x₃) #; AD=AutoReverseDiff())

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300
)

optsol, _ = layer(nothing, uopt + zero(ComponentArray(ps)), st)

f = Figure()
ax = CairoMakie.Axis(f[1, 1])
scatterlines!(ax, optsol, vars = [:x₁, :x₂])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2, 1])
stairs!(ax1, optsol, vars = [:u₁])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
f

## Multiple Shooting
shooting_points = [0.0, 3.0, 6.0, 9.0, 12.0]
mslayer = MultipleShootingLayer(
    prob, Tsit5(), shooting_points...; controls = (1 => control,),
    bounds_p = ([1.0, 1.0], [1.0, 1.0])
)

msps, msst = LuxCore.setup(Random.default_rng(), mslayer)

optprob = OptimizationProblem(
    mslayer, :x₃
)

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-5,
    hessian_approximation = "limited-memory",
    max_iter = 300 # 165
)

blocks = Corleone.get_block_structure(mslayer)

uopt = solve(
    optprob, BlockSQPOpt(),
    opttol = 1.0e-6,
    options = blockSQP.sparse_options(),
    sparsity = blocks,
    maxiters = 300 # 165
)

mssol, _ = mslayer(nothing, uopt + zero(ComponentArray(msps)), msst)

f = Figure()
ax = CairoMakie.Axis(f[1, 1])
scatterlines!(ax, mssol, vars = [:x₁, :x₂])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2, 1])
stairs!(ax1, mssol, vars = [:u₁])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
f
