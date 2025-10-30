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
lb, ub = Corleone.get_bounds(ol)
crit= ACriterion()
ACrit = crit(ol)

sampling = Corleone.get_sampling_constraint(ol)
sampling(p, nothing)

sampling_cons = let sampling=sampling
    (res, p, ::Any) -> begin
        res .= sampling(p, nothing)
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

nc = Corleone.control_blocks(ol)
f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:2:12, title="States")
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:2:12, title="Sensitivities")
ax3 = CairoMakie.Axis(f[2,:], xticks = 0:1:12, title="Sampling")
[plot!(ax, optsol.t, sol) for sol in eachrow(Array(optsol))[1:2]]
[plot!(ax2, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[Corleone.sensitivity_variables(ol)[:]])))]
stairs!(ax3, last(ol.layer.controls).t, (uopt + zero(p)).controls[nc[1]+1:nc[2]])
stairs!(ax3, last(ol.layer.controls).t, (uopt + zero(p)).controls[nc[2]+1:nc[3]])
f

multiplier = uopt.original.inner.mult_g

IG = InformationGain(ol, uopt)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
scatter!(ax, IG.t, tr.(IG.global_information_gain[1]))
CairoMakie.hlines!(ax, multiplier[1:1])

ax1 = CairoMakie.Axis(f[1,2])
scatter!(ax1, IG.t, tr.(IG.global_information_gain[2]))
CairoMakie.hlines!(ax1, multiplier[2:2])
CairoMakie.linkyaxes!(ax1, ax)
f


# Single Shooting
dt = 0.25
oed_layer = Corleone.OEDLayer(prob, Tsit5(); params=[2,3], controls = (control,),
            control_indices = [1], dt = dt)
ps, st = LuxCore.setup(Random.default_rng(), oed_layer)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(oed_layer)
sols, _ = oed_layer(nothing, ps, st)

criterion = crit(oed_layer)
criterion(p, nothing)

sampling = get_sampling_constraint(oed_layer)
sampling_cons = let ax = getaxes(p), sampling=sampling
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= sampling(ps, nothing)
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

nc = Corleone.control_blocks(oed_layer)
f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks=0:2:12, title="States + control")
ax1 = CairoMakie.Axis(f[2,1], xticks=0:2:12, title="Sensitivities")
ax2 = CairoMakie.Axis(f[1,2], xticks=0:2:12, title="FIM")
ax3 = CairoMakie.Axis(f[2,2], xticks=0:2:12, title="Sampling")
[plot!(ax, optsol.t, sol) for sol in eachrow(Array(optsol))[1:2]]
[plot!(ax1, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[Corleone.sensitivity_variables(oed_layer)[:]])))]
[plot!(ax2, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[Corleone.fisher_variables(oed_layer)])))]
stairs!(ax, control.t, (uopt + zero(p)).controls[nc[1]+1:nc[2]])
stairs!(ax3, 0.0:dt:12.0-dt, (uopt + zero(p)).controls[nc[2]+1:nc[3]])
stairs!(ax3, 0.0:dt:12.0-dt, (uopt + zero(p)).controls[nc[3]+1:nc[4]])
f


## Multiple Shooting
shooting_points = [0.0,4.0, 8.0, 12.0]
oed_mslayer = OEDLayer(prob, Tsit5(), shooting_points; params=[2,3], dt=dt,
            control_indices = [1], controls=(control,),
            bounds_nodes = (0.05 * ones(2), 10*ones(2))
            )



oed_msps, oed_msst = LuxCore.setup(Random.default_rng(), oed_mslayer)
# Or use any of the provided Initialization schemes
oed_msps, oed_msst = ForwardSolveInitialization()(Random.default_rng(), oed_mslayer)
oed_msp = ComponentArray(oed_msps)
oed_ms_lb, oed_ms_ub = Corleone.get_bounds(oed_mslayer)
oed_sols, _ = oed_mslayer(nothing, oed_msp, oed_msst)

crit = ACriterion()
criterion = crit(oed_mslayer)
criterion(oed_msp, nothing)

sampling = Corleone.get_sampling_constraint(oed_mslayer)
sampling(oed_msp, nothing)

shooting_constraints = let layer = oed_mslayer, st = oed_msst, ax = getaxes(oed_msp), sampling=sampling, matching = Corleone.get_shooting_constraints(oed_mslayer)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        matching_ = matching(sols, ps)
        sampling_ = sampling(ps, st)
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

blocks = Corleone.get_block_structure(oed_mslayer)

uopt = solve(optprob, BlockSQPOpt(),
    opttol = 1e-6,
    options = blockSQP.sparse_options(),
    sparsity = blocks,
    maxiters = 100
)

sol_u = uopt + zero(oed_msp)
mssol, _ = oed_mslayer(nothing, oed_msp, oed_msst)

nc = Corleone.control_blocks(oed_mslayer)
f = Figure()
ax = CairoMakie.Axis(f[1,1], title="States + control")
ax1 = CairoMakie.Axis(f[2,1], title="Sensitivities")
ax2 = CairoMakie.Axis(f[1,2], title="FIM")
ax3 = CairoMakie.Axis(f[2,2], title="Sampling")
[plot!(ax,  sol.t, Array(sol)[i,:])  for sol in mssol for i in 1:2]
[plot!(ax1, sol.t, Array(sol)[i,:])  for sol in mssol for i in 3:6]
[plot!(ax2, sol.t, Array(sol)[i,:])  for sol in mssol for i in 7:9]
f

[stairs!(ax, c.controls[1].t,  sol_u["layer_$i"].controls[nc[i][1]+1:nc[i][2]], color=:black) for (i,c) in enumerate(oed_mslayer.layer.layers)]
[stairs!(ax3, c.controls[1].t, sol_u["layer_$i"].controls[nc[i][2]+1:nc[i][3]], color=Makie.wong_colors()[1]) for (i,c) in enumerate(oed_mslayer.layer.layers)]
[stairs!(ax3, c.controls[1].t, sol_u["layer_$i"].controls[nc[i][3]+1:nc[i][4]], color=Makie.wong_colors()[2]) for (i,c) in enumerate(oed_mslayer.layer.layers)]

f

IG = InformationGain(oed_mslayer, uopt)
multiplier = uopt.original.inner.mult_g[end-1:end]
multiplier = uopt.original.multiplier[end-1:end]

f = Figure()
ax = CairoMakie.Axis(f[1,1])
scatter!(ax, IG.t, tr.(IG.global_information_gain[1]))
CairoMakie.hlines!(ax, multiplier[1:1])

ax1 = CairoMakie.Axis(f[1,2])
scatter!(ax1, IG.t, tr.(IG.global_information_gain[2]))
CairoMakie.hlines!(ax1, multiplier[2:2])
CairoMakie.linkyaxes!(ax1, ax)
f
