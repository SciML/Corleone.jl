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

### MultiExperiment with ALL FIXED, JUST SAMPLING TIMES

# Either 2 experiments for the same parameters, i.e.,
multi_exp = MultiExperimentLayer{false}(prob, Tsit5(), 2;
    params=[2,3],
    bounds_p=([0.0, 1.0, 1.0], [0.0, 1.0, 1.0]),
    measurements=[
        ControlParameter(collect(0.0:0.25:11.75), controls=ones(48), bounds=(0.0, 1.0)),
        ControlParameter(collect(0.0:0.25:11.75), controls=ones(48), bounds=(0.0, 1.0)),
    ],
    observed=(u, p, t) -> u[1:2]
)

# OR: Experiments targeting single parameters
multi_exp = MultiExperimentLayer{false}(prob, Tsit5(), [[2],[3]];
    bounds_p=([0.0, 1.0, 1.0], [0.0, 1.0, 1.0]),
    measurements=[
        ControlParameter(collect(0.0:0.25:11.75), controls=ones(48), bounds=(0.0, 1.0)),
        ControlParameter(collect(0.0:0.25:11.75), controls=ones(48), bounds=(0.0, 1.0)),
    ],
    observed=(u, p, t) -> u[1:2]
)
ps, st = LuxCore.setup(Random.default_rng(), multi_exp)

optprob = OptimizationProblem(
    multi_exp, ACriterion(); M = zeros(4) .+ 4.0
)

uopt = solve(optprob, Ipopt.Optimizer(),
    hessian_approximation = "limited-memory",
    max_iter = 300
)

sampling_opt = uopt + zero(ComponentArray(ps))
optsol, _ = multi_exp(nothing, sampling_opt, st)

f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:2:12,title="Experiment 1")
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:2:12,title="Experiment 2")
ax3 = CairoMakie.Axis(f[2,1], xticks = 0:2:12)
ax4 = CairoMakie.Axis(f[2,2], xticks = 0:2:12)
plot!(ax, optsol[1], idxs=[1,2])
plot!(ax2, optsol[2], idxs=[1,2])
stairs!(ax3, optsol[1], vars=[:w₁, :w₂])
stairs!(ax4, optsol[2], vars=[:w₁, :w₂])
f


### Now with controls
# Again: Either two experiments for both parameters p₂ and p₃
multi = MultiExperimentLayer{false}(prob, Tsit5(), 2; #[[2,3],[3]];
    params=[2,3],
    bounds_p=([1.0, 1.0], [1.0, 1.0]),
    controls = (1 => control, ),
    measurements=[
        ControlParameter(collect(0.0:0.25:11.75), controls=ones(48), bounds=(0.0, 1.0)),
        ControlParameter(collect(0.0:0.25:11.75), controls=ones(48), bounds=(0.0, 1.0)),
    ],
    observed=(u, p, t) -> u[1:2]
)

# Or one experiment for each parameter (or combinations)
multi = MultiExperimentLayer{false}(prob, Tsit5(), [[2],[3]];
    bounds_p=([1.0, 1.0], [1.0, 1.0]),
    controls = (1 => control, ),
    measurements=[
        ControlParameter(collect(0.0:0.25:11.75), controls=ones(48), bounds=(0.0, 1.0)),
        ControlParameter(collect(0.0:0.25:11.75), controls=ones(48), bounds=(0.0, 1.0)),
    ],
    observed=(u, p, t) -> u[1:2]
)

ps, st = LuxCore.setup(Random.default_rng(), multi)

optprob = OptimizationProblem(
    multi, ACriterion(); M = zeros(4) .+ 4.0
)

uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-7,
     hessian_approximation = "limited-memory",
     max_iter = 300
)

optu = uopt + zero(ComponentArray(ps))

optsol, _ = multi(nothing, optu, st)

f = Figure(size = (800,800))
for i = 1:multi.n_exp
    ax = CairoMakie.Axis(f[1,i], xticks=0:2:12, title="Experiment $i")
    ax1 = CairoMakie.Axis(f[2,i], xticks=0:2:12)
    ax2 = CairoMakie.Axis(f[3,i], xticks=0:2:12)
    ax3 = CairoMakie.Axis(f[4,i], xticks=0:2:12, limits=(nothing, (-0.05,1.05)))
    plot!(ax, optsol[i], idxs=[1,2])
    plot!(ax1, optsol[i], idxs=collect(values(optsol[1].sys.variables))[startswith.(string.(keys(optsol[1].sys.variables)), "G")])
    plot!(ax2, optsol[i], idxs=collect(values(optsol[1].sys.variables))[startswith.(string.(keys(optsol[1].sys.variables)), "F")])
    stairs!(ax, optsol[i], vars=[:p₁], color=:black)
    stairs!(ax3, optsol[i], vars=[:w₁, :w₂])
end
display(f)