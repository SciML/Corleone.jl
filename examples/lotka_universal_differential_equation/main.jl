# ---
# title: Lotka Volterra Universal Differential Equation
# description: Learn an unknown predator-prey interaction from population observations
# category: parameter_estimation
# id: lotka_ude
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# In this tutorial we fit a hybrid mechanistic and neural model to synthetic
# population observations. A universal differential equation (UDE) replaces an
# unknown part of the dynamics with a trainable function. Here the growth and decay
# rates are known, while the predator–prey interaction is learned by a small
# neural network.
#
# !!! info "Prerequisites"
#     The [Getting Started](../getting_started.md) guide and basic ODE modeling.
#     Some familiarity with [Lux.jl](https://lux.csail.mit.edu) helps, too.

# ## Setup
# Lux defines the network and its explicit parameter/state interface. Corleone
# makes the ODE parameters tunable, and ComponentArrays connects the two layouts.
#
# !!! tip "Running the script locally"
#     Run `julia examples/install_deps.jl` once, then execute
#     `julia --project=examples examples/lotka_universal_differential_equation/main.jl`.
using Corleone
using Lux, Random
using OrdinaryDiffEqTsit5
using Optimization, OptimizationLBFGSB
using ForwardDiff, ComponentArrays
using CairoMakie
Makie.inline!()

# ## Synthetic observations
# Generate trajectories of both populations and add Gaussian observation noise
# with standard deviation 0.05. The data-generation model uses the interaction
# `x * y`. We fix the initial populations and the linear growth and decay rates.
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
network = Lux.Chain(
    hidden = Lux.Dense(2 => 5, tanh),
    interaction = Lux.Dense(5 => 1),
)
network_parameters, network_state = Lux.setup(rng, network)

# Start with small Float64 weights and a positive output bias. `pack` supplies a
# flat ODE parameter vector and a constructor that restores Lux's named layers.
initial_network_parameters = ComponentArray{Float64}(network_parameters)
initial_network_parameters .*= 0.1
initial_network_parameters.interaction.bias .= 0.7
p0, rebuild_network = pack(default_vectorizer(), initial_network_parameters)

# Use fixed test-mode state for these stateless Dense layers during integration,
# and reconstruct the current weights at each right-hand-side call.
#
# !!! note "What the closure captures"
#     The closure only captures the network *structure* and its fixed Lux state.
#     Every trainable number reaches the right-hand side through `p`, so Corleone
#     sees all of them as parameters.
hybrid_lotka = let network = network, network_state = Lux.testmode(network_state),
        rebuild_network = rebuild_network
    function (u, p, t)
        output, _ = Lux.apply(network, u, rebuild_network(p), network_state)
        interaction = only(output)^2
        return [u[1] - interaction, interaction - u[2]]
    end
end

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
        collect(eachindex(p0)); kind = Val(:parameter),
        lowerbound = -3.0, upperbound = 3.0
    ) => nothing
)
ps, st = Lux.setup(rng, layer)

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
(; retcode = fit.retcode, initial_loss, final_loss)

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

# Nice work: the hybrid model reproduces the observed orbit closely.
#
# !!! warning "Interpret the learned term with care"
#     A small trajectory error demonstrates a fit to *these* observations. A
#     single orbit does not identify the interaction over the whole state space,
#     so this fit does not establish accuracy for new initial states or future
#     times.
