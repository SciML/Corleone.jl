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

function lotka_dynamics(u, p, t)
    return [u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1];
            -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2];
            (u[1]-1.0)^2 + (u[2] - 1.0)^2]
end

tspan = (0., 12.)
u0 = [0.5, 0.7, 0.]
p0 = [0.0, 1.0, 1.0]

lotka_dynamics(u0, p0, tspan[1])

prob = ODEProblem(lotka_dynamics, u0, tspan, p0)

control = ControlParameter(
    collect(0.0:0.1:11.9), name = :fishing, bounds=(0.0,1.0)
)

# Single Shooting
layer = Corleone.SingleShootingLayer(prob, Tsit5(), controls=(1 => control,), bounds_p = ([1.0, 1.0], [1.0,1.0]);
            )
ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer) .|> ComponentArray

layer(nothing, ps, st)

loss = let layer = layer, st = st, ax = getaxes(p)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        last(sols.u)[3]
    end
end

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
scatterlines!(ax, optsol, vars=[:x₁, :x₂])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2,1])
stairs!(ax1, optsol, vars=[:u₁])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
f

## Multiple Shooting
shooting_points = [0.0, 3.0, 6.0, 9.0, 12.0]
mslayer = MultipleShootingLayer(prob, Tsit5(), shooting_points...; controls = (1 => control,),
                            bounds_p = ([1.0, 1.0], [1.0,1.0]))

msps, msst = LuxCore.setup(Random.default_rng(), mslayer)
msp = ComponentArray(msps)
ms_lb, ms_ub = Corleone.get_bounds(mslayer) .|> ComponentArray

msp[:]
ms_lb[:]

msloss = let layer = mslayer, st = msst, ax = getaxes(msp)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        last(sols.u)[3]
    end
end

shooting_constraints = let layer = mslayer, st = msst, ax = getaxes(msp)
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        Corleone.shooting_constraints!(res, sols)
    end
end

matching = shooting_constraints(zeros(15), msp, nothing)

optfun = OptimizationFunction(
    msloss, AutoForwardDiff(), cons = shooting_constraints
)

optprob = OptimizationProblem(
    optfun, collect(msp), lb = collect(ms_lb), ub = collect(ms_ub), lcons = zero(matching), ucons=zero(matching)
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300 # 165
)

blocks = Corleone.get_block_structure(mslayer)

uopt = solve(optprob, BlockSQPOpt(),
    opttol = 1e-6,
    options = blockSQP.sparse_options(),
    sparsity = blocks,
    maxiters = 300 # 165
)

mssol, _ = mslayer(nothing, uopt + zero(msp), msst)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
scatterlines!(ax, mssol, vars=[:x₁, :x₂])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2,1])
stairs!(ax1, mssol, vars=[:u₁])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
f