using Pkg
Pkg.activate(@__DIR__)
using Corleone
using CorleoneOED
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
    sensealg = SciMLBase.NoAD()
    )
control = ControlParameter(
    collect(0.0:0.25:11.75), name = :fishing, bounds = (0.,1.)
)

# Fixed and without controls
layer = SingleShootingLayer(prob, Tsit5(), bounds_p=([0.0, 1.0, 1.0], [0.0, 1.0, 1.0]))

oed = OEDLayer{false}(
  layer,
  params=[2,3],
  measurements=[
    ControlParameter(collect(0.0:0.25:11.75), controls=ones(48), bounds=(0.0, 1.0)),
    ControlParameter(collect(0.0:0.25:11.75), controls=ones(48), bounds=(0.0, 1.0)),
  ],
  observed=(u, p, t) -> u[1:2],
)

ps, st = LuxCore.setup(Random.default_rng(), oed)

sol, _ = oed(nothing, ps, st)

CorleoneOED._local_information_gain(oed, sol)
CorleoneOED.__fisher_information(oed, sol)
CorleoneOED.get_sampling_sums(oed, sol, ps, st)
CorleoneOED.get_sampling_sums!(zeros(2), oed, sol, ps, st)



optprob = OptimizationProblem(oed, ACriterion(); M =[4.0, 4.0])

uopt = solve(optprob, Ipopt.Optimizer(),
    tol=1e-6,
    hessian_approximation="limited-memory",
    max_iter=50,
    #print_level=3,
)

sol, _  = oed(nothing, uopt.u + zero(ComponentArray(ps)), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:2:12, title="States")
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:2:12, title="Sensitivities")
ax3 = CairoMakie.Axis(f[2,:], xticks = 0:1:12, title="Sampling")
plot!(ax, sol, idxs=[1,2])
plot!(ax2, sol, idxs=[3,4,5,6])
stairs!(ax3, sol, vars=[:w₁, :w₂])
f

# Single Shooting with controls
layer = SingleShootingLayer(prob, Tsit5(), controls = (1 => control, ),
                bounds_p=([1.0, 1.0], [1.0, 1.0]))

oed = OEDLayer{false}(
  layer,
  params=[2,3],
  measurements=[
    ControlParameter(collect(0.0:0.25:11.75), controls=ones(48), bounds=(0.0, 1.0)),
    ControlParameter(collect(0.0:0.25:11.75), controls=ones(48), bounds=(0.0, 1.0)),
  ],
  observed=(u, p, t) -> u[1:2],
)

ps, st = LuxCore.setup(Random.default_rng(), oed)
optprob = OptimizationProblem(oed, ACriterion(); M =[4.0, 4.0])

uopt = solve(optprob, Ipopt.Optimizer(),
    tol=1e-6,
    hessian_approximation="limited-memory",
    max_iter=50,
    #print_level=3,
)

sol, _  = oed(nothing, uopt.u + zero(ComponentArray(ps)), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:2:12, title="States")
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:2:12, title="Sensitivities")
ax3 = CairoMakie.Axis(f[2,1], xticks = 0:1:12, title="Sampling")
ax4 = CairoMakie.Axis(f[2,2], xticks = 0:1:12, title="Controls")
plot!(ax, sol, idxs=[1,2])
plot!(ax2, sol, idxs=[3,4,5,6])
stairs!(ax3, sol, vars=[:w₁, :w₂])
stairs!(ax4, sol, vars=[:p₁])
f

## TODO: Multiple Shooting
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
