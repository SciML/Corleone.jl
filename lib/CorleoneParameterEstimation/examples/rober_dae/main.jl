#src ---
#src title: Parameter Estimation for the Robertson DAE
#src description: Estimate a chosen subset of the rate constants of the classic stiff Robertson DAE — a differential-algebraic system with an algebraic conservation constraint — from noisy observations, using the `params` keyword to pin `k₂` and fit only `k₁` and `k₃`, a Log parameter transform for the strictly positive rates, and the Jacobian-based uncertainty diagnostics that expose the strong residual correlation between the two fitted fast rates
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
#   3. **Estimating a subset of parameters** — the `params` keyword pins `k₂` at its known
#      value and fits only `k₁` and `k₃`, which the Jacobian-based diagnostics still report
#      as strongly correlated.
#
# CorleoneParameterEstimation handles all three: it builds a ParameterEstimationLayer from
# the DAEProblem, uses `params = [:k1, :k3]` to select the free rates and uses a `Log` parameter transform
# (the rates are strictly positive and order-of-magnitude separated), and reports
# covariance/standard errors/confidence intervals from the residual Jacobian at the optimum
# — whose correlation matrix exposes the residual correlation among the fitted rates.

# ## Model
#
# Three reactions, with mole fractions $x_1,x_2,x_3$:
#
#   1. $\mathrm{A} \xrightarrow{k_1} \mathrm{B}$
#   2. $2\mathrm{B} \xrightarrow{k_2} \mathrm{C} + \mathrm{B}$
#   3. $\mathrm{B} + \mathrm{C} \xrightarrow{k_3} \mathrm{A} + \mathrm{C}$
#
# The DAE model reads
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

# ## Sampling and data generation
#
# We sample sparsely on a logarithmic grid for a modest, realistic total of 25 observation 
# times.

t_obs = exp.(range(log(1e-9), log(500.0), length = 25))

sol = solve(prob, DFBDF(); saveat = t_obs)
M = reduce(hcat, sol.u)
σ = 1e-4
M_noisy = M .+ σ .* randn(rng, size(M))
data = ExperimentData(sol.t, M_noisy; σ = σ)

# ## Estimation
#
# Wrap the DAE in a `ParameterEstimationLayer` (starting from a deliberately wrong
# guess with an error of 50%) and fit the rates on a **logarithmic scale** via the `Log` 
# transform, so parameters that differ enormously in scale stay numerically well behaved.
#
# The `params = [:k1, :k3]` keyword (forwarded to the underlying shooting layer) tells the
# layer to treat **only** `k₁` and `k₃` as free degrees of freedom; `k₂` stays fixed at its
# value in `prob.p` (`3.0e7`). The initial guess is therefore drawn only for `k₁` and `k₃`
# (we synthesize the perturbation on-demand by fixing `k₂` in `p_guess`), and the `Log`
# transforms are supplied as **one per free parameter** (`[Log(), Log()]`):
#
p_guess = [0.04 * (1 + 0.5 * randn(rng)), 3.0e7, 1.0e4 * (1 + 0.5 * randn(rng))]
pe = ParameterEstimationLayer(
    remake(prob, p = p_guess), DFBDF(), data; params = [:k1, :k3]
)
pe_log_problem = CorleoneParameterEstimationProblem(pe; transforms = [Log(), Log()])
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
println("Correlation matrix of parameters:      ", est.correlation)
println("Condition number of the residual Jacobian: ", round(est.condition_number; sigdigits = 3))

# All reported quantities are in **natural** units: the covariance is propagated back from
# the log space with the delta method, and the (asymmetric) confidence intervals are formed
# by back-transforming the internal log-space endpoints.
#
# The fit recovers `k₁` and `k₃` to ~4 significant digits and  
# the residual-Jacobian condition number is `≈ 2×10⁶`.
# Still, the **correlation matrix** reports `k₁` and `k₃` correlated at `≈ 0.99`.

# ## Visualizing the fit
#
# Overlay fitted trajectories (lines) and noisy observations (points) via the
# `CorleoneCairoMakieExtension` recipe. The optimizer solves in *log* space, so we replay
# the layer on the **problem** `pe_log_problem` (which carries the `Log` transforms) to get
# a trajectory in natural parameter space — passing the bare `pe` would mistake the log-space
# solution for natural parameters. 
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
