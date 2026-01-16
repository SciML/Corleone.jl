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

oed = OEDLayer{true}(
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
CorleoneOED.fisher_information(oed, sol,ps,st)
CorleoneOED.get_sampling_sums(oed, sol, ps, st)
CorleoneOED.get_sampling_sums!(zeros(2), oed, sol, ps, st)



optprob = OptimizationProblem(oed, ACriterion(); M =[12.0, 12.0])

uopt = solve(optprob, Ipopt.Optimizer(),
    tol=1e-6,
    #hessian_approximation="limited-memory",
    max_iter=5500,
    #print_level=3,
)
#
optu = uopt.u + zero(ComponentArray(ps))
sol, _  = oed(nothing, optu, st)

CorleoneOED.get_sampling_sums(oed, sol, optu, st)

f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:2:12, title="States")
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:2:12, title="Sensitivities")
ax3 = CairoMakie.Axis(f[2,:], xticks = 0:1:12, title="Sampling")
plot!(ax, sol, idxs=[1,2])
plot!(ax2, sol, idxs=[3,4,5,6])
#stairs!(ax3, sol, vars=[:w₁, :w₂])
sampling = [optu.controls[x] for x in st.observation_grid.grid[1:end-1]]
stairs!(ax3, first.(sampling))
stairs!(ax3, last.(sampling))

f

# Single Shooting with controls
layer = SingleShootingLayer(prob, Tsit5(), controls = (1 => control, ),
                bounds_p=([1.0, 1.0], [1.0, 1.0]))

oed = OEDLayer{true}(
  layer,
  params=[2,3],
  measurements=[
    ControlParameter(collect(0.0:0.25:11.75), controls=.5 * ones(48), bounds=(0.0, 1.0)),
    ControlParameter(collect(0.0:0.25:11.75), controls=.5 * ones(48), bounds=(0.0, 1.0)),
  ],
  observed=(u, p, t) -> u[1:2],
)

ps, st = LuxCore.setup(Random.default_rng(), oed)

sol, _ = oed(nothing, ps, st)
sol.sys.variables

optprob = OptimizationProblem(oed, ACriterion(); M =[12.0, 12.0])

CorleoneOED.get_sampling_sums(oed, sol, ps, st)

uopt = solve(optprob, Ipopt.Optimizer(),
    tol=1e-6,
    hessian_approximation="limited-memory",
    max_iter=500,
    #print_level=3,
)

optu = uopt.u + zero(ComponentArray(ps))
sol, _  = oed(nothing, optu, st)

f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:2:12, title="States")
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:2:12, title="Sensitivities")
ax3 = CairoMakie.Axis(f[2,1], xticks = 0:1:12, title="Sampling")
ax4 = CairoMakie.Axis(f[2,2], xticks = 0:1:12, title="Controls")
plot!(ax, sol, idxs=[1,2])
plot!(ax2, sol, idxs=[3,4,5,6])
sampling = [optu.controls[x] for x in st.observation_grid.grid[1:end-1]]
stairs!(ax3, first.(sampling))
stairs!(ax3, last.(sampling))
stairs!(ax4, sol, vars=[:p₁])
f

## TODO: Multiple Shooting
shooting_points = [0.0,6.0]


layer = MultipleShootingLayer(prob, Tsit5(), shooting_points..., controls = (1 => control, ),
                bounds_p=([1.0, 1.0], [1.0, 1.0]))


oed = OEDLayer{false}(
  layer,
  params=[2,3],
  measurements=[
    ControlParameter(collect(0.0:0.25:11.75), controls=.1*ones(48), bounds=(0.0, 1.0)),
    ControlParameter(collect(0.0:0.25:11.75), controls=.15*ones(48), bounds=(0.0, 1.0)),
  ],
  observed=(u, p, t) -> u[1:2],
)

ps, st = LuxCore.setup(Random.default_rng(), oed)

sol, _ = oed(nothing, ps, st)
plot(sol)

CorleoneOED._local_information_gain(oed, sol)
CorleoneOED.__fisher_information(oed, sol)
CorleoneOED.fisher_information(oed, sol,ps,st)
CorleoneOED.get_sampling_sums(oed, sol, ps, st)
CorleoneOED.get_sampling_sums!(zeros(2), oed, sol, ps, st)
Corleone.shooting_constraints(sol)

crit = ACriterion()
crit(oed, nothing, ps, st)

optprob = OptimizationProblem(oed, ACriterion(); M= [4.0,4.0])

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 500
)

blocks = Corleone.get_block_structure(oed.layer)

uopt = solve(optprob, BlockSQPOpt(),
    opttol = 1e-6,
    options = blockSQP.sparse_options(),
    sparsity = blocks,
    maxiters = 100
)

optu = uopt + zero(ComponentArray(ps))
sol, _ = oed(nothing, optu, st)

f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:2:12, title="States")
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:2:12, title="Sensitivities")
ax3 = CairoMakie.Axis(f[2,1], xticks = 0:1:12, title="Sampling")
ax4 = CairoMakie.Axis(f[2,2], xticks = 0:1:12, title="Controls")
plot!(ax, sol, idxs=[1,2])
plot!(ax2, sol, idxs=[3,4,5,6])
stairs!(ax3, sol, vars = [:w₁, :w₂])
stairs!(ax4, sol, vars=[:p₁])
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
