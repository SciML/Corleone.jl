using Corleone
using Lux, Random
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

network = Lux.Chain(
    hidden = Lux.Dense(2 => 5, tanh),
    interaction = Lux.Dense(5 => 1),
)
network_parameters, network_state = Lux.setup(rng, network)

initial_network_parameters = ComponentArray{Float64}(network_parameters)
initial_network_parameters .*= 0.1
initial_network_parameters.interaction.bias .= 0.7
p0, rebuild_network = pack(default_vectorizer(), initial_network_parameters)

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

layer = ShootingLayer(
    hybrid_problem, Tsit5(),
    Tunable(
        collect(eachindex(p0)); kind = Val(:parameter),
        lowerbound = -3.0, upperbound = 3.0
    ) => nothing
)
ps, st = Lux.setup(rng, layer)

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
(; retcode = fit.retcode, initial_loss, final_loss)

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
