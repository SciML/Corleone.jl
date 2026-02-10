#src ---
#src title: The Lotka Volterra Multiexperiment problem
#src description: A beginner-friendly guide to multiexperiments in OED
#src tags:
#src   - Beginner
#src   - Optimal Experimental Design
#src   - Multiexperiments
#src icon: 🌊
#src ---

# ## Setup
# We will use `Corleone` and `CorleoneOED`to model the optimal experimental design problem.
using Corleone, CorleoneOED

# Additionally, we will need the folllowing packages
# - [`LuxCore`]() and [`Random`]() for basic setup functions
# - [`OrdinaryDiffEqTsit5`]() as an adaptive solver for the related ODEProblem
# - [`SymbolicIndexingInterface`]() to conviniently access variables and controls of the solution
# - [`Optimization`](), [`OptimizationMOI`](), [`Ipopt`](), and [`ComponentArrays`]() to setup and solve the optimization problem
# - [`CairoMakie`]() to plot the solution

using LuxCore
using Random
using OrdinaryDiffEqTsit5
using SymbolicIndexingInterface
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using CairoMakie

# ## Lotka Volterrra Dynamics

function lotka_dynamics(du, u, p, t)
    du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    return
end
tspan = (0.0, 12.0)
u0 = [0.5, 0.7]
p0 = [0.0, 1.0, 1.0]
prob = ODEProblem(lotka_dynamics, u0, tspan, p0)

# ## Construct OEDLayer from SingleShootingLayer

cgrid = collect(0.0:0.25:11.75)
control = ControlParameter(
    cgrid, name=:fishing, bounds=(0.0, 1.0)
)

multi_exp = MultiExperimentLayer{false}(
    prob, Tsit5(), 2;
    params = [2, 3],
    bounds_p = ([1.0, 1.0], [1.0, 1.0]),
    controls = (1 => control,),
    measurements = [
        ControlParameter(cgrid, controls = ones(48), bounds = (0.0, 1.0)),
        ControlParameter(cgrid, controls = ones(48), bounds = (0.0, 1.0)),
    ],
    observed = (u, p, t) -> u[1:2]
)

function plot_lotka(sols)
    f = Figure()
    for (i,sol) in enumerate(sols)
        ax = CairoMakie.Axis(f[1, i], xticks = 0:2:12, title = "Experiment $i\nStates")
        ax2 = CairoMakie.Axis(f[2, i], xticks = 0:2:12, title = "Sensitivities")
        ax3 = CairoMakie.Axis(f[3, i], xticks = 0:1:12, title = "Sampling")
        ax4 = CairoMakie.Axis(f[4, i], xticks = 0:1:12, title = "Controls")
        plot!(ax, sol, idxs = [1, 2])
        plot!(ax2, sol, idxs = [3, 4, 5, 6])
        stairs!(ax3, sol, vars=[:w₁, :w₂])
        stairs!(ax4, sol, vars = [:p₁])
    end
    f
end

optprob = OptimizationProblem(
    multi_exp, ACriterion(); M = zeros(4) .+ 4.0
)

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol=1.0e-7,
    hessian_approximation="limited-memory",
    max_iter=100
);

#
ps, st = LuxCore.setup(Random.default_rng(), multi_exp)
optsol, _ = multi_exp(nothing, uopt + zero(ComponentArray(ps)), st)

plot_lotka(optsol)