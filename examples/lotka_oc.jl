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
layer = Corleone.SingleShootingLayer(prob, Tsit5(), [1], (control,);
            # uncomment and adapt the following line if (parts of) u0 need to be optimized as well
            #tunable_ic = [1], bounds_ic = (0.3, 0.9)
            )
ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer)

layer(nothing, ps, st)

loss = let layer = layer, st = st, ax = getaxes(p)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        sols[:x₃][end]
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

## Multiple Shooting
shooting_points = [0.0, 3.0, 6.0, 9.0, 12.0]
mslayer = Corleone.MultipleShootingLayer(prob, Tsit5(),[1], (control,), shooting_points;
            bounds_nodes = ([0.05,0.05, 0.0], 10*ones(3)),
            # uncomment and adapt the following line if (parts of) u0 need to be optimized as well
            #tunable_ic = [1,2], bounds_ic = (.3 * ones(2), .9*ones(2))
            )
msps, msst = LuxCore.setup(Random.default_rng(), mslayer)
# Or use any of the Initialization schemes
msps, msst = ConstantInitialization(Dict(1=>1.0,2=>1.0,3=>1.0))(Random.default_rng(), mslayer)
msps, msst = ForwardSolveInitialization()(Random.default_rng(), mslayer)
msp = ComponentArray(msps)
ms_lb, ms_ub = Corleone.get_bounds(mslayer)

msloss = let layer = mslayer, st = msst, ax = getaxes(msp)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        last(sols)[:x₃][end]
    end
end

shooting_constraints = let layer = mslayer, st = msst, ax = getaxes(msp), matching_constraint = Corleone.get_shooting_constraints(mslayer)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        matching_constraint(sols, ps)
    end
end

matching = shooting_constraints(msp, nothing)
eq_cons(res, x, p) = res .= shooting_constraints(x, p)

optfun = OptimizationFunction(
    msloss, AutoForwardDiff(), cons = eq_cons
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

f = Figure(size = (400,400))
for j in 1:4
    ax = f[j, 1] = CairoMakie.Axis(f)
    for i in 1:4
        plt = i == 4 ? stairs! : lines!
        plt(ax, mssol[i].t, Array(mssol[i])[j, :],)
    end
end
display(f)
