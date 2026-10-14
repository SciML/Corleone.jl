#src ---
#src title: Parameter Estimation for the Robertson DAE
#src description: Estimate the three rate constants of the classic stiff Robertson DAE — a differential-algebraic system with an algebraic conservation constraint — from noisy observations, using a Log parameter transform for the strictly positive rates and showing how the Jacobian-based uncertainty diagnostics expose the near-perfect correlation (and near non-identifiability) between two of the parameters
#src tags:
#src   - Differential-algebraic equation (DAE)
#src   - Stiff problem
#src   - Parameter Estimation
#src   - Log transform
#src   - Uncertainty
#src   - Correlation
#src icon: ⚗️
#src ---

# ## [Parameter estimation for the Robertson DAE](@id corleone_pe_rober_dae)
#
# The classic stiff **Robertson problem** showcases three things in one small model:
#
#   1. **DAE structure** — the third equation is algebraic (a mass-conservation constraint,
#      no time derivative), so the model is an index-1 DAE, not an ODE;
#   2. **Stiffness** — the rate constants span ~nine orders of magnitude, forcing an
#      implicit BDF integrator;
#   3. **Correlated parameters** — two of the three rates are ~perfectly correlated in the
#      data, so the experiment can barely separate them despite an excellent fit.
#
# CorleoneParameterEstimation handles all three: it drives the DAE straight through the
# layer, uses a `Log` parameter transform (the rates are strictly positive and
# order-of-magnitude separated), and reports covariance/standard errors/confidence intervals
# from the residual Jacobian at the optimum — whose correlation matrix and condition number
# expose the near non-identifiability.

# ## Model

# $\mathrm{A} \to \mathrm{B}$ ($k_1$), $2\mathrm{B} \to \mathrm{C}+\mathrm{B}$ ($k_2$),
# $\mathrm{B}+\mathrm{C} \to \mathrm{A}+\mathrm{C}$ ($k_3$), with mole fractions $x_1,x_2,x_3$:
#
# ```math
# \begin{aligned}
# \dot x_1 &= -k_1 x_1 + k_3 x_2 x_3,\\
# \dot x_2 &=  k_1 x_1 - k_3 x_2 x_3 - k_2 x_2^2,\\
# 0 &= x_1 + x_2 + x_3 - 1.
# \end{aligned}
# ```
#
# The last line is the algebraic constraint (marked `differential_vars = false`, no `du` slot).

using Corleone: SciMLBase
using Corleone
using CorleoneParameterEstimation
using OrdinaryDiffEqBDF, OrdinaryDiffEqNonlinearSolve
using ComponentArrays
using LuxCore
using StableRNGs
using SymbolicIndexingInterface
using NonlinearSolve
using CairoMakie

rng = StableRNG(1111)

function robertson!(out, du, u, p, t)
    k₁, k₂, k₃ = p
    x₁, x₂, x₃ = u
    out[1] = -k₁ * x₁ + k₃ * x₂ * x₃ - du[1]
    out[2] = k₁ * x₁ - k₃ * x₂ * x₃ - k₂ * x₂^2 - du[2]
    out[3] = x₁ + x₂ + x₃ - 1   # algebraic: no `du` term
    return
end

du0 = [-0.04, 0.04, 0.0]
u0 = [1.0, 0.0, 0.0]
differential_vars = [true, true, false]
sys = SymbolCache([:x1, :x2, :x3], [:k1, :k2, :k3], :t)
f = DAEFunction(robertson!; sys = sys)

# True rates span nine orders of magnitude; the problem is stiff, so we use BDF (`DFBDF`)
# and give the DAE a consistent `du0` + initialization scheme.
p0 = [0.04, 3.0e7, 1.0e4]
prob = DAEProblem(f, du0, u0, (0.0, 500.0), p0; abstol = 1.0e-8, reltol = 1.0e-8,
    differential_vars = differential_vars, initializealg = BrownFullBasicInit())

# ## Sampling
#
# The dynamics act on two scales: a fast transient near `t = 0`, where species `B` (`x₂`)
# settles to a tiny quasi-steady level under the huge rates `k₂`, `k₃`, and a slow tail,
# where `A` (`x₁`) decays under `k₁` over tens of seconds. We sample densely near `t = 0`
# (logarithmically `1e-10 … 1e0`) and uniformly over the slow tail (`0.5` spacing up to
# `500`). This log-in-time sampling is what lets the fit resolve the two fast rates at all.
t_fast = exp.(range(log(1e-10), log(1e0), length = 300))
t_slow = collect(1.0:0.5:500.0)
t_obs = sort(unique(vcat(t_fast, t_slow)));

sol = solve(prob, DFBDF(); saveat = t_obs)
M = reduce(hcat, sol.u)
σ = 1e-4
M_noisy = M .+ σ .* randn(rng, size(M))
data = ExperimentData(sol.t, M_noisy; σ = σ)

# ## Estimation
#
# Wrap the DAE in a `ParameterEstimationLayer` (starting from a deliberately wrong 50%
# guess) and fit the rates on a **logarithmic scale** via the `Log` transform, so parameters
# that differ enormously in scale stay numerically well behaved:
#
pe = ParameterEstimationLayer(
    remake(prob, p = p0 .* (ones(3) + 0.5 * randn(rng, 3))), DFBDF(), data
)
pe_log_problem = CorleoneParameterEstimationProblem(pe; transforms = [Log(), Log(), Log()])
nlls_prob = NonlinearLeastSquaresProblem(pe_log_problem)

println("Initial parameter guess (in log space): ", nlls_prob.u0)

psol = solve(nlls_prob, LevenbergMarquardt();
    abstol = 1.0e-8, reltol = 1.0e-8, show_trace = Val(false), maxiters = 1000)
println("Fitted parameters (in log space):     ", psol.u)

# ## Results & uncertainty
#
est = estimate_parameters(pe_log_problem, psol)

println("Fitted rate constants (natural scale): ", round.(est.x; sigdigits = 4))
println("True rate constants:                   ", p0)
println("Standard errors:                       ", round.(est.std_errors; sigdigits = 3))
println("95% confidence intervals:              ", est.confidence_intervals)
println("Correlation matrix of parameters:")
display(est.correlation)
println()
println("Condition number of the residual Jacobian: ", round(est.condition_number; sigdigits = 3))

# All reported quantities are in **natural** units: the covariance is propagated back from
# the log space with the delta method, and the (asymmetric) confidence intervals are formed
# by back-transforming the internal log-space endpoints.
#
# The fit is excellent (`k₁` recovered to ~5 significant figures), but the **correlation
# matrix** shows `k₂` and `k₃` correlated at `0.999999` while `k₁` is almost orthogonal to
# both. Both fast rates enter `ẋ₂` as the compensating terms `k₂x₂²` and `k₃x₂x₃`, so a
# larger `k₂` can trade against a smaller `k₃` with almost no change to the prediction. The
# residual Jacobian is correspondingly ill-conditioned (`cond ≈ 10¹³`, far above the `1e8`
# identifiability threshold), so `estimate_parameters` warns that these parameters may be
# non-identifiable. This is *not* a bug, but the model+data honestly reporting that, for
# this experiment, `k₂` and `k₃` can only be pinned down jointly.

# ## Visualizing the fit
#
# Overlay fitted trajectories (lines) and noisy observations (points) via the
# `CorleoneCairoMakieExtension` recipe. The optimizer solves in *log* space, so we replay
# the layer on the **problem** `pe_log_problem` (which carries the `Log` transforms) to get
# a trajectory in natural parameter space — passing the bare `pe` would mistake the log-space
# solution for natural parameters. Time is on a log axis so both regimes are visible.
#
result = fitted_parameters(pe_log_problem, psol)
tick_pos = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e1, 1e2, 5e2]
tick_lab = ["10⁻¹⁰", "10⁻⁸", "10⁻⁶", "10⁻⁴", "10⁻²", "10⁰", "10¹", "10²", "500"]
f = CairoMakie.Figure()
ax = CairoMakie.Axis(f[1, 1];
    xlabel = "Time t", ylabel = "mole fraction",
    xscale = log10, xticks = (tick_pos, tick_lab),
    title = "Robertson DAE fit (log-time)", xticklabelrotation = pi / 4,
)
plot!(ax, result; labels = ["x₁", "x₂", "x₃"])
axislegend(ax; position = :lc)
f
