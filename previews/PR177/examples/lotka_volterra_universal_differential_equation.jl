using Corleone
using LuxCore, Random
using OrdinaryDiffEqTsit5
using Optimization, OptimizationLBFGSB
using ForwardDiff, ComponentArrays
using CairoMakie
Makie.inline!()

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

layer = ShootingLayer(
    hybrid_problem, Tsit5(),
    Tunable(
        collect(1:21); kind = Val(:parameter),
        lowerbound = -3.0, upperbound = 3.0
    ) => nothing
)
ps, st = LuxCore.setup(rng, layer)

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

_, reconstruct = pack(default_vectorizer(), model.ps)
fitted_parameters = reconstruct(fit.u)
fitted, _ = layer(nothing, fitted_parameters, optprob.p)
final_loss = population_loss(fitted)
@assert final_loss < initial_loss / 10
(; initial_loss, final_loss)

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

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
