#src ---
#src title: Compartmental OED Problem
#src description: A simple OED problem with discrete measurements
#src tags:
#src   - Compartmental OED
#src   - Optimal Experimental Design
#src   - Discrete measurements
#src icon: 🎣
#src ---

# ## [Compartmental OED](@id compartmental_oed)
# This is a quick intro based on [Compartmental OED problem](https://mintoc.de/index.php?title=Compartmental_OED).
# The goal here is to design an optimal measurement strategy to determine three parameters
# in a one-dimensional ODE model, where we can directly measure the single state.
# The difference to [the Lotka OED problem](@ref lotka_oed) is that in this example we
# are given 18 possible time points at which measurements can be taken. `CorleoneOED` also
# provides this functionality to model discrete measurements.


# ## Setup
# We will use `Corleone` and `CorleoneOED` to model the optimal experimental design problem.
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

# ## Define the dynamics

function compartmental(du, u, p, t)
    θ₁, θ₂, θ₃ = p
    du[1] = θ₃ * (-θ₁ * exp(-θ₁ * t) + θ₂ * exp(-θ₂ * t))
    return
end

tspan = (0., 50.)
u0 = [0.]
p0 = [0.05884, 4.298, 21.80]
prob = ODEProblem(compartmental, u0, tspan, p0)


# ## Construct the discrete measurement `OEDLayer`

# Constructing the `OEDLayer` with discrete measurement model is done via the specialization
# `{true}`. As there are also no other controls acting on the system and the initial values
# are fixed, this is another special case of OED, i.e., the solution of the differential
# equation is fixed for some parts of the variables, namely the original differential states
# and their sensitivity with respect to the parameters. `CorleoneOED` takes this into account
# to accelerate computations by not integrating these fixed differential states in each
# iteration.

measurement_points = [0.0, 0.166, 0.333, 0.5, 0.666, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 30.0, 48.0]
w_init = 0.5 * ones(length(measurement_points))

oed = OEDLayer{true}(
    prob, Tsit5(),
    params = [1, 2, 3],
    bounds_p=(p0, p0),
    measurements = [
        ControlParameter(measurement_points, controls = w_init, bounds = (0.0, 1.0)),
    ],
    observed = (u, p, t) -> u[1:1],
)

# ## Visualize solution and solve the problem

# Also with discrete measurements, calling the layer performs the integration.

ps, st = LuxCore.setup(Random.default_rng(), oed)

sol, _ = oed(nothing, ps, st)

function plot_oed(sol, sampling)
    f = Figure()
    ax = CairoMakie.Axis(f[1, 1],  xticks = 0:10:50, title = "States")
    ax2 = CairoMakie.Axis(f[2, 1], xticks = 0:10:50, title = "Sensitivities")
    ax3 = CairoMakie.Axis(f[3, 1], xticks = 0:10:50, title = "Sampling")
    plot!(ax, sol, idxs = [1])
    plot!(ax2, sol, idxs = [2, 3, 4])
    scatter!(ax3, measurement_points, sampling)
    f
end

plot_oed(sol, ps.controls)

# We assume now that we can measure at 6 of the 18 possible measurement times and specify
# this upper bound via the keyworded argument `M` when constructing the optimization problem.

optprob = OptimizationProblem(oed, DCriterion(); M =[6.0])

uopt = solve(optprob, Ipopt.Optimizer(),
    tol=1e-12,
    hessian_approximation="limited-memory",
    max_iter=100,
)

optu = uopt.u + zero(ComponentArray(ps))
sol, _  = oed(nothing, optu, st)

plot_oed(sol, optu.controls)
