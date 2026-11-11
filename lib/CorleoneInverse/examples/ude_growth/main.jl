# ---
# title: Universal differential equation with a Lux network
# description: Learn a nonlinear reaction from one experiment and warm-start a joint fit.
# category: parameter_estimation
# id: inverse_ude_growth
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# Adapted from Christoph Plate (@chplate)'s example in
# [Corleone.jl PR #168](https://github.com/SciML/Corleone.jl/pull/168).

# The known dynamics are linear decay. A Lux network learns the unknown reaction
# g(u) = 0.5 sin(u) through the ODE solution, using noisy trajectory observations.
using CorleoneInverse, Lux, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5
using Optimization, OptimizationOptimisers, OptimizationLBFGSB
using StableRNGs, CairoMakie, LinearAlgebra
using Distributions: Laplace

rng = StableRNG(1)
g_true(u) = 0.5 * sin(u)
true_rhs(u, p, t) = [-0.5 * u[1] + g_true(u[1])]
t = collect(0.0:0.2:10.0)
σ = 0.01
function measure(u0)
    truth = solve(ODEProblem(true_rhs, [u0], (0.0, 10.0)), Tsit5(); saveat = t, abstol = 1.0e-9, reltol = 1.0e-9)
    return Experiment(1 => (t, getindex.(truth.u, 1) .+ σ .* randn(rng, length(t)), σ); u0 = [u0])
end
data_1, data_2 = measure(2.0), measure(0.3)

# ## Embed a Lux network
# A 1 → 4 → 2 → 1 network keeps forward-mode training small enough for a tutorial.
# Zero-initialized output weights start the UDE at the known linear decay.
# Rebuild the named network parameters from the flat ODE parameter vector.
network = Lux.Chain(Lux.Dense(1 => 4, tanh), Lux.Dense(4 => 2, tanh), Lux.Dense(2 => 1))
network_ps, network_st = LuxCore.setup(rng, network)
network_ps = ComponentArray{Float64}(network_ps)
network_ps.layer_3.weight .= 0
network_ps.layer_3.bias .= 0
p0, rebuild = pack(default_vectorizer(), network_ps)
network_st = LuxCore.testmode(network_st)
@noinline function ude_rhs(u, p, t)
    output, _ = network(u, rebuild(p), network_st)
    return [-0.5 * u[1] + only(output)]
end
prob = ODEProblem(ude_rhs, [2.0], (0.0, 10.0), p0; abstol = 1.0e-7, reltol = 1.0e-7)
shooting = ShootingLayer(prob, Tsit5())
shared = Tunable(eachindex(p0)) => nothing
first_experiment = Experiments(prob, shared; experiments = (first = data_1,))
ps_1, st_1 = LuxCore.setup(rng, first_experiment)
# Small chunks limit compilation cost for the differentiated Lux/ODE objective.
ad = Optimization.AutoForwardDiff(; chunksize = 1)
model_1 = CorleoneDynamicOptProblem(shooting, first_experiment; ps = ps_1, st = st_1)
opt_1 = OptimizationProblem(model_1, ad)

# ## Adam followed by L-BFGS-B
# Adam takes an unconstrained problem and locates a basin. L-BFGS-B refines
# the trajectory fit.
adam_1 = solve(remake(opt_1; lb = nothing, ub = nothing), Adam(0.01); maxiters = 100)
fit_1 = solve(remake(opt_1; u0 = adam_1.u), LBFGSB(); maxiters = 100)
@assert isfinite(fit_1.objective)
println("First-experiment objective = ", fit_1.objective)

# ## Add a second initial condition
# The low-state experiment constrains a region absent from the first trajectory.
experiments = Experiments(prob, shared; experiments = (first = data_1, second = data_2))
ps, st = LuxCore.setup(rng, experiments)
model = CorleoneDynamicOptProblem(shooting, experiments; ps, st)
opt = OptimizationProblem(model, ad)
adam = solve(remake(opt; u0 = fit_1.u, lb = nothing, ub = nothing), Adam(0.01); maxiters = 100)
fit = solve(remake(opt; u0 = adam.u), LBFGSB(); maxiters = 100)
@assert isfinite(fit.objective)
println("Joint objective = ", fit.objective)

# ## L1 regularization through a prior
# A Laplace prior adds λ * sum(abs, weights) to the scalar objective. Its additive
# normalization constant does not affect the minimizer. Initial values remain fixed.
λ = 1.0e-4
regularized = Experiments(
    prob, Tunable(eachindex(p0); prior = Laplace(0.0, 1 / λ)) => nothing;
    experiments = (first = data_1, second = data_2)
)
ps_l1, st_l1 = LuxCore.setup(rng, regularized)
model_l1 = CorleoneDynamicOptProblem(shooting, regularized; ps = ps_l1, st = st_l1)
opt_l1 = OptimizationProblem(model_l1, ad)
fit_l1 = solve(remake(opt_l1; u0 = fit.u), LBFGSB(); maxiters = 100)
@assert isfinite(fit_l1.objective)
println("Regularized weight norm = ", norm(fit_l1.u))

# ## Compare trajectories and learned reaction terms
_, reconstruct = pack(default_vectorizer(), ps)
trajectories, _ = shooting(experiments, reconstruct(fit.u), st)
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
axislegend(reaction_axis)
figure
