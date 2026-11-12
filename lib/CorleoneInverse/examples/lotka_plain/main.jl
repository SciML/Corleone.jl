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

# The classical predator–prey model has parameters `p = [α, β, δ, γ]`.
# One noisy experiment determines the four rates while its initial state is known.
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
prob = ODEProblem(lotka_volterra!, [1.0, 1.0], (0.0, 10.0), p_true; abstol = 1.0e-9, reltol = 1.0e-9)
t = collect(0.0:0.1:10.0)
truth = solve(prob, Tsit5(); saveat = t)
y = reduce(hcat, truth.u) .+ 0.05 .* randn(rng, 2, length(t))
observations = Experiment(1 => (t, y[1, :], 0.05), 2 => (t, y[2, :], 0.05))

# ## Fit in internal coordinates
# The positive transform is declared on each annotation. Lux setup maps the natural
# starting guesses to internal coordinates, and ParameterEstimate maps results back.
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
println(estimate)
@assert maximum(abs.(estimate.x ./ p_true .- 1)) < 0.15

# Known noise scales give weighted residuals and covariance `(JᵀJ)⁻¹` in internal
# coordinates. The delta method propagates covariance to natural coordinates.
_, reconstruct = pack(default_vectorizer(), ps)
trajectory, _ = layer(observations, reconstruct(fit.u), experiment_state(layer, observations, st))
figure = Figure()
axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "Population")
for i in 1:2
    lines!(axis, trajectory.t, getindex.(trajectory.u, i); label = "Fitted species $i")
    scatter!(axis, t, y[i, :]; markersize = 4, label = "Observed species $i")
end
axislegend(axis)
figure

# ## Unknown initial conditions
# An annotation at t = 0 adds an initial value to the tunable tree.
ic_layer = ShootingLayer(
    prob, Tsit5(), Tunable(1:4; transform = asℝ₊) => nothing,
    Tunable(1:2; kind = Val(:variable), transform = asℝ₊) => [0.0]
)
ic_ps, ic_st = LuxCore.setup(rng, ic_layer)
@assert length(NonlinearLeastSquaresProblem(CorleoneDynamicOptProblem(ic_layer, observations; ps = ic_ps, st = ic_st)).u0) == 6

# ## Optimization.jl interface
# The same observations also define a scalar least-squares objective for Ipopt.
opt = OptimizationProblem(model, Optimization.AutoForwardDiff())
ipopt_fit = solve(opt, Ipopt.Optimizer(); print_level = 0, tol = 1.0e-8)
@assert SciMLBase.successful_retcode(ipopt_fit)
println(ParameterEstimate(model, ipopt_fit.u))
figure
