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
using LinearAlgebra

function lotka_dynamics(u, p, t)
    return [u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1];
            -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]]
end

tspan = (0., 12.)
u0 = [0.5, 0.7, ]
p0 = [0.0, 1.0, 1.0]
prob = ODEProblem{false}(lotka_dynamics, u0, tspan, p0,
    sensealg = ForwardDiffSensitivity()
    )
control = ControlParameter(
    collect(0.0:0.25:11.75), name = :fishing, bounds = (0.,1.)
)

# Single Shooting
layer = CorleoneCore.SingleShootingLayer(prob, Tsit5(),[1], (control,); tunable_ic = [1,2], bounds_ic=([0.3,0.3], [0.9,0.9]))
oed_layer = CorleoneCore.augment_layer_for_oed(layer; observed = (u,p,t) -> u[1:2])
ps, st = LuxCore.setup(Random.default_rng(), oed_layer)
p = ComponentArray(ps)
lb, ub = CorleoneCore.get_bounds(oed_layer)
nc, dt = length(control.t), diff(control.t)[1]

sols, _ = oed_layer(nothing, ps, st)
Acrit = ACriterion()
Acrit(oed_layer, sols)
Ecrit = ECriterion()
Ecrit(oed_layer, sols)

loss = let layer = oed_layer, st = st, ax = getaxes(p), crit= DCriterion()
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        crit(layer, sols)
    end
end

loss(collect(p), nothing)

sampling_cons = let layer = oed_layer, st = st, ax = getaxes(p)
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= [sum(ps.controls[nc+1:2*nc]) * dt;
          sum(ps.controls[2*nc+1:3*nc]) * dt  ]
    end
end

sampling_cons(zeros(2),collect(p), nothing)

optfun = OptimizationFunction(
    loss, AutoForwardDiff(), cons = sampling_cons
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub), lcons=zeros(2), ucons=[4.0, 4.0]
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300
)

optsol, _ = oed_layer(nothing, uopt + zero(p), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
ax1 = CairoMakie.Axis(f[2,1])
ax2 = CairoMakie.Axis(f[1,2])
ax3 = CairoMakie.Axis(f[2,2])
[plot!(ax, optsol.t, sol) for sol in eachrow(Array(optsol))[1:2]]
[plot!(ax1, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[CorleoneCore.sensitivity_variables(oed_layer)])))]
[plot!(ax2, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[CorleoneCore.fisher_variables(oed_layer)])))]
stairs!(ax, control.t, (uopt + zero(p)).controls[1:length(control.t)])
stairs!(ax3, control.t, (uopt + zero(p)).controls[length(control.t)+1:2*length(control.t)])
stairs!(ax3, control.t, (uopt + zero(p)).controls[2*length(control.t)+1:3*length(control.t)])
f


## Multiple Shooting
shooting_points = [0.0,4.0, 8.0, 12.0]
mslayer = CorleoneCore.MultipleShootingLayer(prob, Tsit5(),[1], (control,), shooting_points;
        bounds_nodes = (0.05 * ones(2), 10*ones(2)))#, tunable_ic = [1,2], bounds_ic = ([.3,.3], [.9,.9]))
msps, msst = LuxCore.setup(Random.default_rng(), mslayer)
msp = ComponentArray(msps)
lb, ub = CorleoneCore.get_bounds(mslayer)

mslayer(nothing, msps, msst)
mslayer(nothing, lb, msst)

# + OED
oed_mslayer = CorleoneCore.augment_layer_for_oed(mslayer; observed = (u,p,t) -> u[1:2])
oed_msps, oed_msst = LuxCore.setup(Random.default_rng(), oed_mslayer)
# Or use any of the provided Initialization schemes
oed_msps, oed_msst = ForwardSolveInitialization()(Random.default_rng(), oed_mslayer)
oed_msp = ComponentArray(oed_msps)
oed_ms_lb, oed_ms_ub = CorleoneCore.get_bounds(oed_mslayer)
oed_sols, _ = oed_mslayer(nothing, oed_msp, oed_msst)

msloss = let layer = oed_mslayer, st = oed_msst, ax = getaxes(oed_msp), crit=ACriterion()
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        crit(layer, sols)
    end
end


msloss(oed_msp, nothing)
nc_ms = length(first(oed_mslayer.layers).controls[1].t)
shooting_constraints = let layer = oed_mslayer, st = oed_msst, ax = getaxes(oed_msp), matching_constraint = CorleoneCore.get_shooting_constraints(oed_mslayer)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        matching_ = matching_constraint(sols, ps)
        sampling_ = [sum(reduce(vcat, [ps["layer_$i"].controls[nc_ms+1:2*nc_ms] for i in 1:length(layer.layers)])) * dt;
                    sum(reduce(vcat, [ps["layer_$i"].controls[2*nc_ms+1:3*nc_ms] for i in 1:length(layer.layers)])) *  dt]
        return vcat(matching_, sampling_)
    end
end

matching = shooting_constraints(oed_msp, nothing)
jac_cons = ForwardDiff.jacobian(Base.Fix2(shooting_constraints, nothing), oed_msp)
eq_cons(res, x, p) = res .= shooting_constraints(x, p)

optfun = OptimizationFunction(
    msloss, AutoForwardDiff(), cons = eq_cons
)

ucons = zero(matching)
ucons[end-1:end] .= 4.0
optprob = OptimizationProblem(
    optfun, collect(oed_msp), lb = collect(oed_ms_lb), ub = collect(oed_ms_ub), lcons = zero(matching), ucons=ucons
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300 # 165
)

blocks = CorleoneCore.get_block_structure(oed_mslayer)

uopt = solve(optprob, BlockSQPOpt(),
    opttol = 1e-6,
    options = blockSQP.sparse_options(),
    sparsity = blocks,
    maxiters = 300 # 165
)


sol_u = uopt + zero(oed_msp)
lc = first(oed_mslayer.layers).controls[1].t |> length

mssol, _ = oed_mslayer(nothing, oed_msp, oed_msst)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
ax1 = CairoMakie.Axis(f[2,1])
ax2 = CairoMakie.Axis(f[1,2])
ax3 = CairoMakie.Axis(f[2,2])
[plot!(ax,  sol.t, Array(sol)[i,:])  for sol in mssol for i in 1:2]
[plot!(ax1, sol.t, Array(sol)[i,:])  for sol in mssol for i in 3:6]
[plot!(ax2, sol.t, Array(sol)[i,:])  for sol in mssol for i in 7:9]
f

[stairs!(ax, c.controls[1].t,  sol_u["layer_$i"].controls[1:lc], color=:black) for (i,c) in enumerate(mslayer.layers)]
[stairs!(ax3, c.controls[1].t, sol_u["layer_$i"].controls[lc+1:2*lc], color=Makie.wong_colors()[1]) for (i,c) in enumerate(mslayer.layers)]
[stairs!(ax3, c.controls[1].t, sol_u["layer_$i"].controls[2*lc+1:3*lc], color=Makie.wong_colors()[2]) for (i,c) in enumerate(mslayer.layers)]

f