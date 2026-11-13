# ---
# title: Parameter estimation for the Robertson DAE
# description: Estimate two positive rates in a stiff differential-algebraic reaction model.
# category: parameter_estimation
# id: inverse_rober_dae
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# Adapted from Christoph Plate (@chplate)'s example in
# [Corleone.jl PR #168](https://github.com/SciML/Corleone.jl/pull/168).

# Robertson's stiff reaction kinetics couple two differential equations to mass
# conservation, giving a differential-algebraic equation (DAE). The rate ``k_2`` is
# known, while ``k_1`` and ``k_3`` are estimated in logarithmic coordinates.
#
# !!! info "What you will learn"
#     How to fit a subset of the parameters of a `DAEProblem` and keep the others
#     `Constant`, using symbols instead of indices.
using Corleone: SciMLBase
using CorleoneInverse, LuxCore, ComponentArrays, NonlinearSolve, StableRNGs
using OrdinaryDiffEqBDF, OrdinaryDiffEqNonlinearSolve, CairoMakie, LinearAlgebra
using SymbolicIndexingInterface: SymbolCache
using TransformVariables: asℝ₊

function robertson!(out, du, u, p, t)
    k₁, k₂, k₃ = p
    x₁, x₂, x₃ = u
    out[1] = -k₁ * x₁ + k₃ * x₂ * x₃ - du[1]
    out[2] = k₁ * x₁ - k₃ * x₂ * x₃ - k₂ * x₂^2 - du[2]
    out[3] = x₁ + x₂ + x₃ - 1
    return nothing
end
rng = StableRNG(1111)
p_true = [0.04, 3.0e7, 1.0e4]
sys = SymbolCache([:x1, :x2, :x3], [:k1, :k2, :k3], :t)
prob = DAEProblem(
    DAEFunction(robertson!; sys), [-0.04, 0.04, 0.0], [1.0, 0.0, 0.0], (0.0, 500.0),
    p_true; differential_vars = [true, true, false],
    initializealg = BrownFullBasicInit(), abstol = 1.0e-9, reltol = 1.0e-9
)

# The dynamics span many time scales, so the measurement times are spaced
# logarithmically.
t = exp.(range(log(1.0e-9), log(500.0); length = 25))
t[end] = last(prob.tspan)
truth = solve(prob, DFBDF(); saveat = t)
@assert SciMLBase.successful_retcode(truth)
σ = 1.0e-4
y = reduce(hcat, truth.u) .+ σ .* randn(rng, 3, length(t))
data = Experiment((i => (t, y[i, :], σ) for i in 1:3)...)

# ## Fit two of the three rates
# The `SymbolCache` lets us refer to parameters by name. `k2` is `Constant` at its
# true value; the two other rates start from perturbed guesses.
layer = ShootingLayer(
    prob, DFBDF(),
    Tunable(:k1; default = 0.06, transform = asℝ₊) => nothing,
    Constant(:k2; default = p_true[2]) => nothing,
    Tunable(:k3; default = 1.5e4, transform = asℝ₊) => nothing
)
ps, st = LuxCore.setup(rng, layer)
model = CorleoneDynamicOptProblem(layer, data; ps, st)
fit = solve(NonlinearLeastSquaresProblem(model), LevenbergMarquardt(); maxiters = 1000)
@assert SciMLBase.successful_retcode(fit)
estimate = ParameterEstimate(model, fit.u)
@assert isapprox(estimate.x, p_true[[1, 3]]; rtol = 0.1)
estimate

# The correlation matrix shows how strongly the two estimates are coupled.
correlation = estimate.covariance ./ (estimate.std_errors * estimate.std_errors')

# !!! tip "Stiff problems"
#     Choose a stiff DAE solver such as `DFBDF` and tight tolerances. The
#     derivatives are propagated through the implicit solver automatically.
_, reconstruct = pack(default_vectorizer(), ps)
trajectory, _ = layer(data, reconstruct(fit.u), experiment_state(layer, data, st))
figure = Figure()
axis = CairoMakie.Axis(
    figure[1, 1]; xlabel = "Time", ylabel = "Mole fraction", xscale = log10
)
positive_times = findall(>(0), trajectory.t)
for i in 1:3
    lines!(
        axis, trajectory.t[positive_times], getindex.(trajectory.u[positive_times], i);
        label = "x$i"
    )
    scatter!(axis, t, y[i, :]; markersize = 5)
end
axislegend(axis)
figure
