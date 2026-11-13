# ---
# title: Parameter estimation for the Lotka–Volterra ODE
# description: Fit four predator–prey parameters and report asymptotic uncertainties.
# category: parameter_estimation
# id: inverse_lotka_plain
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# Adapted from Christoph Plate (@chplate)'s example in
# [Corleone.jl PR #168](https://github.com/SciML/Corleone.jl/pull/168).

# Let's estimate the four rates `p = [α, β, δ, γ]` of the classical predator–prey
# model from one noisy experiment with a known initial state. Along the way we
# report asymptotic uncertainties and solve the same problem with two different
# solvers.
#
# !!! info "What you will learn"
#     How to turn observations into an [`Experiment`](@ref), declare positive
#     parameters with a transform, and summarize a fit with `ParameterEstimate`.
using Corleone: SciMLBase
using CorleoneInverse, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5, NonlinearSolve
using StableRNGs, CairoMakie
using Optimization, OptimizationMOI, Ipopt
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
prob = ODEProblem(
    lotka_volterra!, [1.0, 1.0], (0.0, 10.0), p_true; abstol = 1.0e-9, reltol = 1.0e-9
)
t = collect(0.0:0.1:10.0)
truth = solve(prob, Tsit5(); saveat = t)
y = reduce(hcat, truth.u) .+ 0.05 .* randn(rng, 2, length(t))

# Each observation pair maps a state index to `(times, values, σ)`, where `σ` is
# the known noise standard deviation.
observations = Experiment(1 => (t, y[1, :], 0.05), 2 => (t, y[2, :], 0.05))

# ## Fit in internal coordinates
# The positive transform `asℝ₊` is declared on the annotation. Lux setup maps the
# natural starting guesses to internal (logarithmic) coordinates, and
# `ParameterEstimate` maps the results back. We start from a perturbed guess.
layer = ShootingLayer(
    remake(prob; p = p_true .+ 0.05 .* randn(rng, 4)), Tsit5(),
    Tunable(1:4; transform = asℝ₊) => nothing
)
ps, st = LuxCore.setup(rng, layer)
model = CorleoneDynamicOptProblem(layer, observations; ps, st)
nlls = NonlinearLeastSquaresProblem(model)
fit = solve(nlls, LevenbergMarquardt(); abstol = 1.0e-8, reltol = 1.0e-8)
@assert SciMLBase.successful_retcode(fit)
estimate = ParameterEstimate(model, fit.u)
@assert maximum(abs.(estimate.x ./ p_true .- 1)) < 0.15
estimate

# !!! note "Where the uncertainties come from"
#     Known noise scales give weighted residuals and the covariance `(JᵀJ)⁻¹` in
#     internal coordinates. The delta method propagates it to natural coordinates.

# Compare the fitted trajectory with the data.
_, reconstruct = pack(default_vectorizer(), ps)
trajectory, _ = layer(
    observations, reconstruct(fit.u), experiment_state(layer, observations, st)
)
figure = Figure()
axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "Population")
for i in 1:2
    lines!(axis, trajectory.t, getindex.(trajectory.u, i); label = "Fitted species $i")
    scatter!(axis, t, y[i, :]; markersize = 4, label = "Observed species $i")
end
axislegend(axis)
figure

# ## Unknown initial conditions
# A variable annotation at ``t = 0`` adds the initial values to the unknowns. The
# problem now has six of them.
ic_layer = ShootingLayer(
    prob, Tsit5(), Tunable(1:4; transform = asℝ₊) => nothing,
    Tunable(1:2; kind = Val(:variable), transform = asℝ₊) => [0.0]
)
ic_ps, ic_st = LuxCore.setup(rng, ic_layer)
ic_model = CorleoneDynamicOptProblem(ic_layer, observations; ps = ic_ps, st = ic_st)
ic_fit = solve(NonlinearLeastSquaresProblem(ic_model), LevenbergMarquardt())
@assert length(ic_fit.u) == 6
ParameterEstimate(ic_model, ic_fit.u)

# ## Optimization.jl interface
# The same model also defines a scalar least-squares objective, so any
# Optimization.jl solver works, too. Here we use Ipopt.
opt = OptimizationProblem(model, Optimization.AutoForwardDiff())
ipopt_fit = solve(opt, Ipopt.Optimizer(); print_level = 0, tol = 1.0e-8)
@assert SciMLBase.successful_retcode(ipopt_fit)
ParameterEstimate(model, ipopt_fit.u)

# !!! tip "Which interface should I use?"
#     `NonlinearLeastSquaresProblem` exploits the least-squares structure and is
#     usually the fastest choice. Use `OptimizationProblem` when you need general
#     priors, constraints, or multiple shooting.
