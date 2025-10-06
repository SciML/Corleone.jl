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

# Single Shooting with fixed controls and fixed u0
ol = OEDLayer(prob, Tsit5(); params= [2,3], dt = 0.2)
ps, st = LuxCore.setup(Random.default_rng(), ol)
p = ComponentArray(ps)
lb, ub = CorleoneCore.get_bounds(ol)
crit= ACriterion()
ACrit = crit(ol)

nc = (0,length(ol.layer.controls[1].t),2*length(ol.layer.controls[1].t))

sampling_cons = let ax = getaxes(p), nc = nc, dt = first(diff(ol.layer.controls[1].t))
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= [sum(ps.controls[nc[i]+1:nc[i+1]]) * dt for i in eachindex(nc)[1:end-1]]
    end
end

optfun = OptimizationFunction(
    crit(ol), AutoForwardDiff(), cons = sampling_cons
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub), lcons=zeros(2), ucons=[4.0, 4.0]
)

uopt = solve(optprob, Ipopt.Optimizer(),
     #tol = 1e-10,
     hessian_approximation = "limited-memory",
     max_iter = 300
)

optsol, _ = ol(nothing, uopt + zero(p), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:2:12, title="States")
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:2:12, title="Sensitivities")
ax3 = CairoMakie.Axis(f[2,:], xticks = 0:1:12, title="Sampling")
[plot!(ax, optsol.t, sol) for sol in eachrow(Array(optsol))[1:2]]
[plot!(ax2, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[CorleoneCore.sensitivity_variables(ol)])))]
stairs!(ax3, last(ol.layer.controls).t, (uopt + zero(p)).controls[nc[1]+1:nc[2]])
stairs!(ax3, last(ol.layer.controls).t, (uopt + zero(p)).controls[nc[2]+1:nc[3]])
f


# Single Shooting
oed_layer = CorleoneCore.OEDLayer(prob, Tsit5(); params=[2,3], controls = (control,),
            control_indices = [1], dt = 0.25)
ps, st = LuxCore.setup(Random.default_rng(), oed_layer)
p = ComponentArray(ps)
lb, ub = CorleoneCore.get_bounds(oed_layer)
nc, dt = length(control.t), diff(control.t)[1]

sols, _ = oed_layer(nothing, ps, st)

criterion = crit(oed_layer)
criterion(p, nothing)

sampling_cons = let layer = oed_layer.layer, st = st, ax = getaxes(p)
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= [sum(ps.controls[nc+1:2*nc]) * dt;
          sum(ps.controls[2*nc+1:3*nc]) * dt  ]
    end
end

sampling_cons(zeros(2),collect(p), nothing)

optfun = OptimizationFunction(
    criterion, AutoForwardDiff(), cons = sampling_cons
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
ax = CairoMakie.Axis(f[1,1], xticks=0:2:12, title="States + control")
ax1 = CairoMakie.Axis(f[2,1], xticks=0:2:12, title="Sensitivities")
ax2 = CairoMakie.Axis(f[1,2], xticks=0:2:12, title="FIM")
ax3 = CairoMakie.Axis(f[2,2], xticks=0:2:12, title="Sampling")
[plot!(ax, optsol.t, sol) for sol in eachrow(Array(optsol))[1:2]]
[plot!(ax1, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[CorleoneCore.sensitivity_variables(oed_layer)])))]
[plot!(ax2, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[CorleoneCore.fisher_variables(oed_layer)])))]
stairs!(ax, control.t, (uopt + zero(p)).controls[1:length(control.t)])
stairs!(ax3, control.t, (uopt + zero(p)).controls[length(control.t)+1:2*length(control.t)])
stairs!(ax3, control.t, (uopt + zero(p)).controls[2*length(control.t)+1:3*length(control.t)])
f


## Multiple Shooting
shooting_points = [0.0,4.0, 8.0, 12.0]
oed_mslayer = OEDLayer(prob, Tsit5(), shooting_points; params=[2,3], dt = 0.25,
            control_indices = [1], controls=(control,),
            bounds_nodes = (0.05 * ones(2), 10*ones(2)))


oed_msps, oed_msst = LuxCore.setup(Random.default_rng(), oed_mslayer)
# Or use any of the provided Initialization schemes
oed_msps, oed_msst = ForwardSolveInitialization()(Random.default_rng(), oed_mslayer)
oed_msp = ComponentArray(oed_msps)
oed_ms_lb, oed_ms_ub = CorleoneCore.get_bounds(oed_mslayer)
oed_sols, _ = oed_mslayer(nothing, oed_msp, oed_msst)

crit = ACriterion()
criterion = crit(oed_mslayer)
criterion(oed_msp, nothing)

nc_ms = length(first(oed_mslayer.layer.layers).controls[1].t)
shooting_constraints = let layer = oed_mslayer, dt = 0.25, st = oed_msst, ax = getaxes(oed_msp), matching_constraint = CorleoneCore.get_shooting_constraints(oed_mslayer)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        matching_ = matching_constraint(sols, ps)
        sampling_ = [sum(reduce(vcat, [ps["layer_$i"].controls[nc_ms+1:2*nc_ms] for i in 1:length(layer.layer.layers)])) * dt;
                    sum(reduce(vcat, [ps["layer_$i"].controls[2*nc_ms+1:3*nc_ms] for i in 1:length(layer.layer.layers)])) *  dt]
        return vcat(matching_, sampling_)
    end
end

eq_cons(res, x, p) = res .= shooting_constraints(x, p)

optfun = OptimizationFunction(
    criterion, AutoForwardDiff(), cons = eq_cons
)
constraints_eval = shooting_constraints(oed_msp, nothing)
ucons = zero(constraints_eval)
ucons[end-1:end] .= 4.0
optprob = OptimizationProblem(
    optfun, collect(oed_msp), lb = collect(oed_ms_lb), ub = collect(oed_ms_ub), lcons = zero(constraints_eval), ucons=ucons
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 100
)

blocks = CorleoneCore.get_block_structure(oed_mslayer)

uopt = solve(optprob, BlockSQPOpt(),
    opttol = 1e-6,
    options = blockSQP.sparse_options(),
    sparsity = blocks,
    maxiters = 100
)


sol_u = uopt + zero(oed_msp)
lc = first(oed_mslayer.layer.layers).controls[1].t |> length

mssol, _ = oed_mslayer(nothing, oed_msp, oed_msst)

f = Figure()
ax = CairoMakie.Axis(f[1,1], title="States + control")
ax1 = CairoMakie.Axis(f[2,1], title="Sensitivities")
ax2 = CairoMakie.Axis(f[1,2], title="FIM")
ax3 = CairoMakie.Axis(f[2,2], title="Sampling")
[plot!(ax,  sol.t, Array(sol)[i,:])  for sol in mssol for i in 1:2]
[plot!(ax1, sol.t, Array(sol)[i,:])  for sol in mssol for i in 3:6]
[plot!(ax2, sol.t, Array(sol)[i,:])  for sol in mssol for i in 7:9]
f

[stairs!(ax, c.controls[1].t,  sol_u["layer_$i"].controls[1:lc], color=:black) for (i,c) in enumerate(oed_mslayer.layer.layers)]
[stairs!(ax3, c.controls[1].t, sol_u["layer_$i"].controls[lc+1:2*lc], color=Makie.wong_colors()[1]) for (i,c) in enumerate(oed_mslayer.layer.layers)]
[stairs!(ax3, c.controls[1].t, sol_u["layer_$i"].controls[2*lc+1:3*lc], color=Makie.wong_colors()[2]) for (i,c) in enumerate(oed_mslayer.layer.layers)]

f