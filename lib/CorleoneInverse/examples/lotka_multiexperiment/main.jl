# ---
# title: Joint parameter estimation over multiple experiments
# description: Combine two predator–prey experiments with shared rates and different initial states.
# category: parameter_estimation
# id: inverse_lotka_multiexperiment
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# Adapted from Christoph Plate (@chplate)'s example in
# [Corleone.jl PR #168](https://github.com/SciML/Corleone.jl/pull/168).

# Two experiments start from different initial conditions and share four model rates.
# Compare a first-experiment fit with a warm-started joint fit.
using Corleone: SciMLBase
using CorleoneInverse, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5, NonlinearSolve
using StableRNGs, CairoMakie
using TransformVariables: asℝ₊

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
    return nothing
end
rng = StableRNG(42)
p_true = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(lotka_volterra!, [1.0, 1.0], (0.0, 10.0), p_true; abstol = 1.0e-9, reltol = 1.0e-9)
t = collect(0.0:0.1:10.0)
function measure(u0)
    truth = solve(remake(prob; u0), Tsit5(); saveat = t)
    y = reduce(hcat, truth.u) .+ 0.05 .* randn(rng, 2, length(t))
    return Experiment(1 => (t, y[1, :], 0.05), 2 => (t, y[2, :], 0.05); u0)
end
data_1, data_2 = measure([1.0, 1.0]), measure([0.4, 2.5])
shooting = ShootingLayer(prob, Tsit5())
shared = Tunable(1:4; default = [1.4, 0.95, 2.8, 0.95], transform = asℝ₊) => nothing
first_experiment = Experiments(prob, shared; experiments = (first = data_1,))
ps_1, st_1 = LuxCore.setup(rng, first_experiment)
model_1 = CorleoneDynamicOptProblem(shooting, first_experiment; ps = ps_1, st = st_1)
fit_1 = solve(NonlinearLeastSquaresProblem(model_1), LevenbergMarquardt())
@assert SciMLBase.successful_retcode(fit_1)
estimate_1 = ParameterEstimate(model_1, fit_1.u)

# ## Add the second experiment
# Initial values belong to each experiment. The tunable rates belong to `shared`.
experiments = Experiments(prob, shared; experiments = (first = data_1, second = data_2))
ps, st = LuxCore.setup(rng, experiments)
_, reconstruct = pack(default_vectorizer(), ps)
warm_start = reconstruct(fit_1.u)
model = CorleoneDynamicOptProblem(shooting, experiments; ps = warm_start, st)
fit = solve(NonlinearLeastSquaresProblem(model), LevenbergMarquardt())
@assert SciMLBase.successful_retcode(fit)
estimate = ParameterEstimate(model, fit.u)
println("Single-experiment standard errors = ", estimate_1.std_errors)
println("Joint standard errors = ", estimate.std_errors)
@assert all(estimate.std_errors .< estimate_1.std_errors)
trajectories, _ = shooting(experiments, reconstruct(fit.u), st)
@assert first(trajectories[2].u) == [0.4, 2.5]

figure = Figure()
for (i, (trajectory, data)) in enumerate(zip(trajectories, (data_1, data_2)))
    axis = CairoMakie.Axis(figure[1, i]; xlabel = "Time", ylabel = "Population", title = "Experiment $i")
    for (j, observation) in enumerate(data.observations)
        lines!(axis, trajectory.t, getindex.(trajectory.u, j); label = "Fitted species $j")
        scatter!(axis, observation.t, observation.y; markersize = 4)
    end
    axislegend(axis)
end
figure
