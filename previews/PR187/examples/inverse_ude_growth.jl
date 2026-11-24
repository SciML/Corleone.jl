using CorleoneInverse, Lux, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5
using NonlinearSolve: LevenbergMarquardt, NonlinearFunction, NonlinearLeastSquaresProblem
using Optimization, OptimizationLBFGSB
using StableRNGs, CairoMakie, LinearAlgebra
using Distributions: Laplace, logpdf

rng = StableRNG(1)
g_true(u) = 0.5 * sin(u)
true_rhs(u, p, t) = [-0.5 * u[1] + g_true(u[1])]
t = collect(0.0:0.2:10.0)
σ = 0.01
function measure(u0, t)
    truth = solve(ODEProblem(true_rhs, [u0], (0.0, 10.0)), Tsit5(); saveat = t, abstol = 1.0e-9, reltol = 1.0e-9)
    return Experiment(1 => (t, getindex.(truth.u, 1) .+ σ .* randn(rng, length(t)), σ); u0 = [u0])
end
data_1, data_2 = measure(2.0, t), measure(0.3, t)

network = Lux.Chain(Lux.Dense(1 => 4, tanh), Lux.Dense(4 => 2, tanh), Lux.Dense(2 => 1))
network_ps, network_st = LuxCore.setup(rng, network)
network_ps = ComponentArray{Float64}(network_ps)
network_ps.layer_3.weight .= 0
network_ps.layer_3.bias .= 0
p0, rebuild = pack(default_vectorizer(), network_ps)
network_st = LuxCore.testmode(network_st)
ude_rhs = let network = network, rebuild = rebuild, network_st = network_st
    (u, p, t) -> begin
        output, _ = network(u, rebuild(p), network_st)
        [-0.5 * u[1] + only(output)]
    end
end
prob = ODEProblem(ude_rhs, [2.0], (0.0, 10.0), p0; abstol = 1.0e-7, reltol = 1.0e-7)

function predict(prob, experiment, weights)
    times = only(experiment.observations).t
    return solve(
        remake(prob; p = weights, u0 = experiment.u0), Tsit5();
        saveat = times, save_everystep = false, dense = false
    )
end

function ude_residual(prob, experiments, weights)
    result = Vector{eltype(weights)}(undef, sum(n_observations, experiments))
    row = 0
    for experiment in experiments
        observation = only(experiment.observations)
        prediction = predict(prob, experiment, weights)
        for i in eachindex(observation.y)
            row += 1
            result[row] = (observation.y[i] - prediction.u[i][1]) / observation.σ
        end
    end
    return result
end

struct UDEResidual
    problem::Any
end

function (residual::UDEResidual)(weights::AbstractVector{T}, experiments) where {T}
    return ude_residual(residual.problem, experiments, weights)::Vector{T}
end

residual_callback = UDEResidual(prob)

function least_squares(f, weights, experiments)
    prototype = zeros(eltype(weights), sum(n_observations, experiments))
    return NonlinearLeastSquaresProblem(
        NonlinearFunction{false}(f; resid_prototype = prototype), weights, experiments
    )
end

ad = Optimization.AutoForwardDiff()
alg = LevenbergMarquardt(; autodiff = ad)
single = least_squares(residual_callback, p0, [data_1])
fit_1 = solve(single, alg; maxiters = 100)
@assert all(isfinite, fit_1.resid)
println("First-experiment mean squared residual = ", sum(abs2, fit_1.resid) / length(fit_1.resid))

experiments = push!(copy(single.p), data_2)
joint = least_squares(residual_callback, fit_1.u, experiments)
fit = solve(joint, alg; maxiters = 100)
@assert all(isfinite, fit.resid)
println("Joint mean squared residual = ", sum(abs2, fit.resid) / length(fit.resid))

λ = 1.0e-4
regularized_objective = let residual_callback = residual_callback, prior = Laplace(0.0, 1 / λ)
    (weights, data) -> begin
        r = residual_callback(weights, data)
        sum(abs2, r) / length(r) - sum(w -> logpdf(prior, w), weights)
    end
end
opt_l1 = OptimizationProblem(OptimizationFunction(regularized_objective, ad), fit.u, experiments)
fit_l1 = solve(opt_l1, LBFGSB(); maxiters = 100)
@assert isfinite(fit_l1.objective)
println("Regularized weight norm = ", norm(fit_l1.u))
println("Regularized objective = ", fit_l1.objective)

trajectories = map(data -> predict(prob, data, fit.u), experiments)
figure = Figure(size = (1100, 440))
axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "State", title = "Joint fit")
for (i, (trajectory, data)) in enumerate(zip(trajectories, (data_1, data_2)))
    observation = only(data.observations)
    lines!(axis, trajectory.t, getindex.(trajectory.u, 1); label = "Experiment $i")
    scatter!(axis, observation.t, observation.y; markersize = 5)
end
axislegend(axis)
reaction_axis = CairoMakie.Axis(figure[1, 2]; xlabel = "State", ylabel = "Reaction", title = "Learned reaction")
ugrid = collect(0.0:0.05:2.0)
lines!(reaction_axis, ugrid, g_true.(ugrid); color = :black, linestyle = :dash, label = "True reaction")
for (weights, label) in ((fit_1.u, "First experiment"), (fit.u, "Joint"), (fit_l1.u, "Joint with L1 prior"))
    values = [only(first(network([u], rebuild(weights), network_st))) for u in ugrid]
    lines!(reaction_axis, ugrid, values; label)
end
axislegend(reaction_axis, position = :rb)
figure

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
