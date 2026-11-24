using Corleone: SciMLBase
using CorleoneInverse, Lux, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5, NonlinearSolve
using Optimization, OptimizationLBFGSB
using StableRNGs, CairoMakie, LinearAlgebra
using SymbolicIndexingInterface: parameter_values_at_time
using Distributions: Laplace

rng = StableRNG(1)
g_true(u) = 0.5 * sin(u)
true_rhs(u, p, t) = [-p[1] * u[1] + g_true(u[1])]
t = collect(0.0:0.2:10.0)
σ = 0.01
function measure(u0)
    truth = solve(
        ODEProblem(true_rhs, [u0], (0.0, 10.0), [0.5]), Tsit5();
        saveat = t, abstol = 1.0e-9, reltol = 1.0e-9
    )
    y = getindex.(truth.u, 1) .+ σ .* randn(rng, length(t))
    return Experiment(1 => (t, y, σ); u0 = [u0])
end
data_1, data_2 = measure(2.0), measure(0.3)

network = Lux.Chain(Lux.Dense(1 => 4, tanh), Lux.Dense(4 => 2, tanh), Lux.Dense(2 => 1))
network_ps, network_st = LuxCore.setup(rng, network)
network_ps = ComponentArray{Float64}(network_ps)
network_ps.layer_3.weight .= 0
network_ps.layer_3.bias .= 0
weights0, rebuild = pack(default_vectorizer(), network_ps)
network_st = LuxCore.testmode(network_st)
n_weights = length(weights0)

ude_rhs = let network = network, rebuild = rebuild, network_st = network_st, n = n_weights
    function (u, p, t)
        output, _ = network(u, rebuild(view(p, 1:n)), network_st)
        k = p[n + 1]
        return [-k * u[1] + only(output)]
    end
end
prob = ODEProblem(
    ude_rhs, [2.0], (0.0, 10.0), [weights0; 0.5]; abstol = 1.0e-7, reltol = 1.0e-7
)

weights = Tunable(1:n_weights) => nothing
decay = Constant(n_weights + 1) => nothing
shooting = ShootingLayer(prob, Tsit5())

first_only = Experiments(prob, weights, decay; experiments = (first = data_1,))
ps_1, st_1 = LuxCore.setup(rng, first_only)
model_1 = CorleoneDynamicOptProblem(shooting, first_only; ps = ps_1, st = st_1)
fit_1 = solve(NonlinearLeastSquaresProblem(model_1), LevenbergMarquardt(); maxiters = 100)
@assert all(isfinite, fit_1.resid)
mean_squared(r) = sum(abs2, r) / length(r)
(; retcode = fit_1.retcode, mean_squared_residual = mean_squared(fit_1.resid))

joint = Experiments(prob, weights, decay; experiments = (first = data_1, second = data_2))
ps, st = LuxCore.setup(rng, joint)
_, reconstruct = pack(default_vectorizer(), ps)
model = CorleoneDynamicOptProblem(shooting, joint; ps = reconstruct(fit_1.u), st)
fit = solve(NonlinearLeastSquaresProblem(model), LevenbergMarquardt(); maxiters = 100)
@assert all(isfinite, fit.resid)
(; retcode = fit.retcode, mean_squared_residual = mean_squared(fit.resid))

λ = 1.0e-4
sparse_weights = Tunable(1:n_weights; prior = Laplace(0.0, 1 / λ)) => nothing
regularized = Experiments(
    prob, sparse_weights, decay; experiments = (first = data_1, second = data_2)
)
_, st_l1 = LuxCore.setup(rng, regularized)
model_l1 = CorleoneDynamicOptProblem(
    shooting, regularized; ps = reconstruct(fit.u), st = st_l1
)
fit_l1 = solve(OptimizationProblem(model_l1, AutoForwardDiff()), LBFGSB(); maxiters = 100)
@assert isfinite(fit_l1.objective)
(; retcode = fit_l1.retcode, objective = fit_l1.objective, weight_norm = norm(fit_l1.u))

function simulate(experiments, template, state, q)
    _, restore = pack(default_vectorizer(), template)
    trajectories, _ = shooting(experiments, restore(q), state)
    trajectory = first(trajectories)
    θ = parameter_values_at_time(trajectory, trajectory, 0.0)[1:n_weights]
    return θ, trajectories
end
_, trajectories = simulate(joint, ps, st, fit.u)
learned_reaction(θ) = u -> only(first(network([u], rebuild(θ), network_st)))

figure = Figure(size = (1100, 440))
axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "State", title = "Joint fit")
for (i, (trajectory, data)) in enumerate(zip(trajectories, (data_1, data_2)))
    observation = only(data.observations)
    lines!(axis, trajectory.t, getindex.(trajectory.u, 1); label = "Experiment $i")
    scatter!(axis, observation.t, observation.y; markersize = 5)
end
axislegend(axis)
reaction_axis = CairoMakie.Axis(
    figure[1, 2]; xlabel = "State", ylabel = "Reaction", title = "Learned reaction"
)
ugrid = collect(0.0:0.05:2.0)
lines!(
    reaction_axis, ugrid, g_true.(ugrid);
    color = :black, linestyle = :dash, label = "True reaction"
)
fits = (
    (first_only, ps_1, st_1, fit_1.u, "First experiment"),
    (joint, ps, st, fit.u, "Joint"),
    (regularized, ps, st_l1, fit_l1.u, "Joint with L1 prior"),
)
for (experiments, template, state, q, label) in fits
    θ, _ = simulate(experiments, template, state, q)
    lines!(reaction_axis, ugrid, learned_reaction(θ).(ugrid); label)
end
axislegend(reaction_axis, position = :rb)
figure

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
