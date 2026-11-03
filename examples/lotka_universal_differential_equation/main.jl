#src ---
#src title: Lotka Volterra Universal Differential Equation
#src description: Learn an unknown predator-prey interaction from population observations
#src icon: 🧠
#src category: parameter_estimation
#src difficulty: Intermediate
#src cover_figure: 1
#src tags:
#src   - Universal Differential Equations
#src   - Parameter Estimation
#src   - Neural Networks
#src ---

# # [Lotka–Volterra Universal Differential Equation](@id lotka_ude)
# Fit a hybrid mechanistic and neural model to synthetic population observations.
# Prerequisites: [Getting Started](../getting_started.md) and basic ODE modeling.
# A universal differential equation (UDE) replaces an unknown part of the dynamics
# with a trainable function. Here growth and decay rates are known, while the
# predator-prey interaction is represented by a small neural network.

# ## Setup
# Run `julia examples/install_deps.jl` once, then execute this script with
# `julia --project=examples examples/lotka_universal_differential_equation/main.jl`.
# The network below uses ordinary array operations, so LuxCore is sufficient;
# no additional neural-network package is needed.
using Corleone
using LuxCore, Random
using OrdinaryDiffEqTsit5
using Optimization, OptimizationLBFGSB
using ForwardDiff, ComponentArrays
using CairoMakie
Makie.inline!()

# ## Synthetic observations
# Generate noiseless observations of both populations. Only this data-generation
# model knows that the interaction is `x * y`. The training model never uses that
# formula. We fix the initial populations and the linear growth and decay rates.
function true_lotka(u, p, t)
    x, y = u
    interaction = x * y
    return [x - interaction, interaction - y]
end

rng = MersenneTwister(42)

observation_times = collect(0.0:0.1:5.0)
u0 = [0.8, 1.2]
tspan = (first(observation_times), last(observation_times))
truth_problem = ODEProblem(true_lotka, u0, tspan)
truth = solve(
    truth_problem, Tsit5(); saveat = observation_times,
    abstol = 1.0e-9, reltol = 1.0e-9
)
@assert successful_retcode(truth)
observations = Array(truth)
observations .+= 0.05 * randn(rng, size(observations))

# ## Hybrid dynamics
# Use a 2 → 5 → 1 network with tanh hidden activations. Store its 21 weights and
# biases in a flat ODE parameter vector. Squaring the output makes the learned
# interaction nonnegative. The same interaction removes prey and adds predators:
#
# ```math
# \dot{x} = x - N_\theta(x,y), \qquad
# \dot{y} = N_\theta(x,y) - y.
# ```
# This sign structure shares mechanistic information between the two equations.
# It does not enforce positive populations: the network need not vanish on the
# axes. Training and interpretation below are restricted to the observed orbit.
function neural_interaction(u, p)
    weights_in = reshape(view(p, 1:10), 5, 2)
    bias_hidden = view(p, 11:15)
    weights_out = view(p, 16:20)
    hidden = tanh.(weights_in * u .+ bias_hidden)
    return (sum(weights_out .* hidden) + p[21])^2
end

function hybrid_lotka(u, p, t)
    interaction = neural_interaction(u, p)
    return [u[1] - interaction, interaction - u[2]]
end

p0 = 0.1 .* randn(rng, 21)

hybrid_problem = ODEProblem(
    hybrid_lotka, u0, tspan, p0;
    saveat = observation_times, abstol = 1.0e-8, reltol = 1.0e-8
)

# All network parameters are tunable and constant over time (`nothing` is the
# switching grid). Bounds keep trial network weights in a moderate range. The
# initial populations remain fixed at `u0` from the ODE problem.
layer = ShootingLayer(
    hybrid_problem, Tsit5(),
    Tunable(
        collect(1:21); kind = Val(:parameter),
        lowerbound = -3.0, upperbound = 3.0
    ) => nothing
)
ps, st = LuxCore.setup(rng, layer)

# ## Fit the population trajectories
# `saveat` fixes the observation grid; no interior resets are introduced. Compare
# saved states directly rather than interpolating Corleone's trajectory object.
function population_loss(trajectory)
    @assert trajectory.t ≈ observation_times
    return sum(
        sum(abs2, trajectory.u[i] .- observations[:, i])
            for i in eachindex(observation_times)
    ) / length(observations)
end

model = CorleoneDynamicOptProblem(layer, population_loss; ps, st)
initial_loss = optimization_objective(model)
optprob = OptimizationProblem(model, AutoForwardDiff())
fit = solve(optprob, OptimizationLBFGSB.LBFGSB(); maxiters = 1_000)
@assert successful_retcode(fit)

# ForwardDiff differentiates through the ODE solve and neural network. Reconstruct
# the structured Lux parameters from the optimizer's packed vector for prediction.
_, reconstruct = pack(default_vectorizer(), model.ps)
fitted_parameters = reconstruct(fit.u)
fitted, _ = layer(nothing, fitted_parameters, optprob.p)
final_loss = population_loss(fitted)
@assert final_loss < initial_loss / 10
(; initial_loss, final_loss)

# ## Compare observations and fit
# `trajectoryplot` draws the trained UDE states (`u₁`: prey, `u₂`: predator).
# Overlay the synthetic observations on its population axis, matching state colors.
figure = trajectoryplot(fitted; show_legend = false, state_string = "Population")
axis = content(figure[1, 1][1, 1])
for (i, label) in ((1, "Prey"), (2, "Predator"))
    scatter!(
        axis, observation_times, observations[i, :]; color = Cycled(i),
        markersize = 6, label = "$label observations"
    )
end
plot!(axis, truth, linestyle = :dash)
figure[1, 2] = Legend(figure[1, 2], axis, framevisible = false)
figure

# A small trajectory error demonstrates a fit to these observations. A single
# orbit does not identify the interaction over the whole state space, and this
# training fit does not establish accuracy for new initial states or future times.
