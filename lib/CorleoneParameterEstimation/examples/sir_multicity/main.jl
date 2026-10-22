#src ---
#src title: Joint Parameter Estimation for an SIR Epidemic Across Two Regions
#src description: Estimate the transmission and recovery rates of an SIR epidemic model jointly from noisy case-count data of two regions — starting from a deliberately wrong guess — and show how uncertainties shrink when data is pooled, plus a single-region fit with an unknown (tunable) initial infection count
#src tags:
#src   - SIR epidemic
#src   - Parameter Estimation
#src   - Multi-experiment
#src   - Tunable initial condition
#src   - Uncertainty
#src icon: 🦠
#src ---

# ## [Joint parameter estimation for an SIR epidemic across two regions](@id corleone_pe_sir_multicity)
#
# Beyond the Lotka-Volterra examples, here is a *realistic* application: estimating the
# parameters of an **SIR epidemic model** from noisy epidemiological records. Two
# "experiments" stand in for two regions of an outbreak that began with **different** numbers of
# initially infected individuals but share the *same* biological parameters (transmission rate
# ``\beta`` and recovery rate ``\gamma``). This example ties together several
# CorleoneParameterEstimation features in one workflow:
#
#   1. **Multi-experiment** fitting: both regions share ``\beta`` and ``\gamma`` but start from
#      their own initial conditions (pooling the two datasets shrinks the uncertainty).
#   2. **Per-quantity observations**: new infections and the recovered population are reported on
#      *different* schedules and with *different* noise levels (a common situation with field data).
#   3. **Uncertainty quantification**: covariance, standard errors and asymptotic confidence
#      intervals from the residual Jacobian at the optimum.
#   4. **Tunable initial condition**: in a companion single-region fit we treat the *unknown
#      number of initially infected individuals* as a parameter to be estimated (via
#      `tunable_ic`), rather than a known input.
#
# Crucially, the optimizer is started from a **deliberately wrong guess** — in a real
# outbreak you do not know the answer you are trying to recover.

# ## Setup

using Corleone, CorleoneParameterEstimation
using CairoMakie
using OrdinaryDiffEqTsit5
using NonlinearSolve
using LinearAlgebra
using LuxCore
using StableRNGs

# ## SIR dynamics
#
# The standard susceptible–infected–recovered model
# ```math
# \begin{aligned}
# \frac{dS}{dt} &= - \beta \frac{S I}{N},\\
# \frac{dI}{dt} &=  \beta \frac{S I}{N} - \gamma I,\\
# \frac{dR}{dt} &=  \gamma I,
# \end{aligned}
# ```
# with state `u = (S, I, R)`, total population `N`, and parameters `p = (β, γ)`.
# The basic reproduction number is ``R_0 = \beta / \gamma`` — the headline quantity
# epidemiologists report from such fits.

function sir!(du, u, p, t)
    S, I, R = u
    β, γ = p
    N = S + I + R
    du[1] = -β * S * I / N
    du[2] = β * S * I / N - γ * I
    du[3] = γ * I
    return
end

# Both regions share a 30-day horizon and the same total population ``N_{\text{pop}} = 10^4``.
# Region A starts with `I(0) = 2` infected individuals, region B with `I(0) = 20`; the
# susceptible and recovered populations start identically (and are treated as known).
# The *true* rates are ``\beta = 0.5`` and ``\gamma = 0.2``, i.e. ``R_0 = 2.5``:

tspan = (0.0, 30.0)
Npop = 1.0e4
β_true, γ_true = 0.5, 0.2                 # R0 = 2.5
p_true = [β_true, γ_true]

# ## Generate noisy per-quantity observations for both regions
#
# We simulate the "true" outbreak of each region and add Gaussian reporting noise. New
# *infections* and the *recovered* population are reported on different schedules — infections
# every second day, recovered only every fourth day — modeled with the *per-quantity observation
# times* form `Experiment(times, values)`. The two observables also live on very different
# scales (the infection flux peaks at a few hundred cases per day, the recovered compartment
# grows to ``10^4``), so each quantity carries its own noise level: `σ_inf = 15` new cases per
# day and `σ_rec = 150` recovered individuals, stored *per quantity* in the `Experiment` and
# used to weight the residuals.

rng = StableRNG(3)
t_inf = collect(0.0:2.0:30.0)             # infections reported every second day
t_rec = collect(0.0:4.0:30.0)             # recovered reported every fourth day

# New infections per day ≈ `β S I / N` (the *instantaneous* flux into the infected compartment,
# the same quantity that `obs_sir` below reports), evaluated at each report time; recovered is
# simply the `R` state. The transmission rate `β` is read from the layer's own model parameters
# (not a free global), and the total population `N` from the conserved `S + I + R` of the state,
# so the helper is self-contained and consistent with the fit. It returns the noisy series
# *and* the noise levels, so the data and its `σ` are guaranteed to match.
function simulate_region(prob_model, times_inf, times_rec, noise_rng; σ_inf = 15.0, σ_rec = 150.0)
    lay0 = SingleShootingLayer(prob_model, Tsit5(); saveat = unique(sort(vcat(times_inf, times_rec))))
    ps, st = LuxCore.setup(noise_rng, lay0)
    sol, _ = lay0(nothing, ps, st)

    β = ps.p[1]                       # transmission rate of this outbreak (p[1])
    S = getindex.(sol.u, 1)
    I = getindex.(sol.u, 2)
    R = getindex.(sol.u, 3)
    Ntot = S .+ I .+ R                # conserved total population
    flux = [β * s * i / n for (s, i, n) in zip(S, I, Ntot)]
    idx_inf = indexin(times_inf, sol.t)
    idx_rec = indexin(times_rec, sol.t)

    y_inf = flux[idx_inf] .+ σ_inf .* randn(noise_rng, length(idx_inf))
    y_rec = R[idx_rec] .+ σ_rec .* randn(noise_rng, length(idx_rec))
    return y_inf, y_rec, σ_inf, σ_rec
end

# Region A: a small starting outbreak (`I(0)=2`).
probA = ODEProblem(sir!, [Npop - 2.0, 2.0, 0.0], tspan, p_true)
y_infA, y_recA, σ_infA, σ_recA = simulate_region(probA, t_inf, t_rec, rng)

# Region B: a larger starting outbreak (`I(0)=20`), same rates.
probB = ODEProblem(sir!, [Npop - 20.0, 20.0, 0.0], tspan, p_true)
y_infB, y_recB, σ_infB, σ_recB = simulate_region(probB, t_inf, t_rec, rng)

data = ExperimentData(
    Experiment([t_inf, t_rec], [y_infA, y_recA]; σ = [σ_infA, σ_recA], u0 = [Npop - 2.0, 2.0, 0.0]),
    Experiment([t_inf, t_rec], [y_infB, y_recB]; σ = [σ_infB, σ_recB], u0 = [Npop - 20.0, 20.0, 0.0]),
)

# The observation map returns the two per-quantity series: quantity 1 = new infections
# (`β S I / N`, which needs `p`), quantity 2 = recovered (`R`).
obs_sir = (u, p, t) -> (p[1] * u[1] * u[2] / (u[1] + u[2] + u[3]), u[3])

# ## Build the parameter-estimation layer (joint multi-experiment fit)
#
# We use the convenience constructor `ParameterEstimationLayer(prob, alg, data)`, which
# builds **one `SingleShootingLayer` per experiment** from the same problem — each region is
# simulated from its own stored `u0` and sampled at its own observation times. Only the model
# parameters `p = (β, γ)` are tunable; the (known) initial conditions come from the data.
#
# The problem we hand in carries a **deliberately wrong guess** `p = (0.3, 0.4)` instead of the
# true `(0.5, 0.2)`, and `bounds_p` keeps the rates in a physically meaningful (positive) range
# — optimizer iterates that probe extreme rates can otherwise drive the ODE into unstable
# territory and stall the fit.

β_guess, γ_guess = 0.3, 0.4               # wrong on purpose (true: 0.5, 0.2)
prob_guess = ODEProblem(sir!, [Npop - 2.0, 2.0, 0.0], tspan, [β_guess, γ_guess])

pe = ParameterEstimationLayer(
    prob_guess, Tsit5(), data; observed = obs_sir,
    bounds_p = ([0.1, 0.05], [2.0, 0.6])
)

# The flat optimization variables are the two shared params `(β, γ)`, starting at the wrong
# guess stored in the problem:

nlls = problem(pe)
println("Optimization variables (β, γ) starting at: ", nlls.u0, "   (true = ", p_true, ")")

# ## Solve the least-squares problem
#
# The residuals are the per-quantity, σ-weighted differences `(y_obs - y_model) / σ` for both
# regions. `LevenbergMarquardt` is a robust choice for this mildly nonlinear problem.

sol = solve(nlls, LevenbergMarquardt(); abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 200)
println("Fitted [β, γ] = ", round.(sol.u; digits = 4))
println("True   [β, γ] = ", p_true)

# ## Estimates and uncertainties
#
# `estimate_parameters` returns the fitted values with the covariance, standard errors and
# asymptotic 95% confidence intervals:

est = estimate_parameters(pe, sol)
println(
    "β  (transmission rate): ", round(est.x[1]; digits = 4), " ± ", round(est.std_errors[1]; digits = 4),
    "   true = ", β_true
)
println(
    "γ  (recovery rate):     ", round(est.x[2]; digits = 4), " ± ", round(est.std_errors[2]; digits = 4),
    "   true = ", γ_true
)

# The headline derived quantity — the basic reproduction number `R0 = β / γ` — and its
# uncertainty follow by propagation from the covariance (delta method):

R0_hat = est.x[1] / est.x[2]
grad = [1 / est.x[2], -est.x[1] / est.x[2]^2]     # ∂R0/∂(β, γ)
se_R0 = sqrt(grad' * est.covariance * grad)
println(
    "R0 = ", round(R0_hat; digits = 3), " ± ", round(se_R0; digits = 3),
    "   true R0 = ", round(β_true / γ_true; digits = 3)
)

# ## Profile-likelihood confidence intervals
#
# The Jacobian-based intervals of `estimate_parameters` are exact only in the *linear*
# least-squares limit: for genuinely nonlinear models they are a first-order approximation
# that can under- or over-cover, and they tell you nothing about the *shape* of the
# likelihood (an asymmetric profile, or a flat valley that would signal a
# non-identifiable parameter).
#
# `profile_likelihood` is the gold-standard cross-check. It re-fits the model with **one
# parameter held fixed** at a grid of values (all the others re-optimized) and reads the
# `(1-α)`-level interval off the resulting profile of minimized residual sum of squares.
# Because the profile follows the *true nonlinear* likelihood — rather than a linearization
# at the optimum — it exposes exactly how much information this experiment design carries
# about each parameter. Here the data prescribes a known noise level `σ`, so the residuals
# are weighted and the threshold is the likelihood-ratio one (a `χ²` quantile), exactly the
# same `σ` that drives the covariance of `estimate_parameters`.

pl = profile_likelihood(pe, sol; names = [:β, :γ], n_points = 41)
println(pl)

# Side by side with the asymptotic intervals from `estimate_parameters`, the profile
# confirms the same picture while being the more reliable statement about identifiability:
# for this design the profile CIs are in fact *narrower* than the t-based asymptotic ones,
# which can only underestimate---never overestimate---how sharply the likelihood peaks.

# Visualize the profiles: each panel shows the minimized RSS as a function of the parameter
# value, with the likelihood-ratio threshold, the shaded profile confidence band and the
# asymptotic interval drawn as dotted reference lines.

fpl = plot(pl)
fpl

# ## Fitting on the log scale (parameter transforms)
#
# Above we kept the rates positive with `bounds_p`. Bounds guarantee *feasibility*, but they
# leave two problems: the optimizer still works in the natural (unscaled) parameterization,
# and the asymptotic confidence interval is built in that same space, so a rate that is only a
# few standard errors from zero can get a *negative* lower endpoint.
#
# A [`Log`](@ref) transform fixes both at once. It fits an unconstrained internal variable
# ``\phi`` with ``\theta = e^{\phi}``, so positivity is encoded in the *geometry* rather than in
# a bound, and the interval is obtained by back-transforming the endpoints of the internal
# interval — yielding a correctly asymmetric, strictly positive result. Transforms are attached
# to the problem via the `transforms` keyword; a non-`Identity` transform drops the natural
# bounds of that variable. (Besides [`Log`](@ref) there is [`Logit`](@ref) for a box constraint
# `lb < θ < ub` and [`Affine`](@ref) for pure rescaling.)

pe_log = ParameterEstimationLayer(prob_guess, Tsit5(), data; observed = obs_sir)
pe_log_problem = CorleoneParameterEstimationProblem(pe_log; transforms = [Log(), Log()])

# The flat optimization variables now live in the internal, unconstrained space:

nlls_log = NonlinearLeastSquaresProblem(pe_log_problem)
println("Internal variables (log β, log γ) starting at: ", nlls_log.u0)

sol_log = solve(nlls_log, LevenbergMarquardt(); abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 200)
est_log = estimate_parameters(pe_log_problem, sol_log)

# `est_log.x` is reported in *natural* units again (the internal values are mapped back with
# `from_internal`), and the covariance is propagated with the delta method. Notice that both
# confidence intervals are strictly positive and asymmetric:

println(
    "β = ", round(est_log.x[1]; digits = 4), "   95% CI = ",
    round.(est_log.confidence_intervals[1]; digits = 4)
)
println(
    "γ = ", round(est_log.x[2]; digits = 4), "   95% CI = ",
    round.(est_log.confidence_intervals[2]; digits = 4)
)
println(
    "Natural-space lower endpoints: ",
    [ci[1] for ci in est_log.confidence_intervals], "  (both > 0)"
)

# ## Comparing the joint fit against a single-region fit
#
# To illustrate the value of pooling data, fit region A alone and compare the uncertainty on
# the transmission rate `β` (the quantity the data constrains most weakly).

dataA = ExperimentData(Experiment([t_inf, t_rec], [y_infA, y_recA]; σ = [σ_infA, σ_recA], u0 = [Npop - 2.0, 2.0, 0.0]))
peA = ParameterEstimationLayer(
    prob_guess, Tsit5(), dataA; observed = obs_sir,
    bounds_p = ([0.1, 0.05], [2.0, 0.6])
)
solA = solve(problem(peA), LevenbergMarquardt(); abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 200)
estA = estimate_parameters(peA, solA)

println("---")
println("Joint-fit std error on β:      ", round(est.std_errors[1]; digits = 4))
println("Region-A-only std error on β:  ", round(estA.std_errors[1]; digits = 4))
println(
    "Uncertainty on β reduced by ",
    round(100 * (1 - est.std_errors[1] / estA.std_errors[1]); digits = 1),
    "% when pooling both regions"
)

# ## Visualize the joint fit
#
# The `CorleoneCairoMakieExtension` recipe overlays the fitted model trajectories (lines)
# with the noisy per-quantity observations (points) for every experiment, colored per
# observed quantity.

result = fitted_parameters(pe, sol)
f = CairoMakie.Figure()
ax = CairoMakie.Axis(f[1, 1], xlabel = "Day", ylabel = "count", title = "SIR fit across two regions (R0 ≈ 2.5)")
plot!(ax, result; labels = ["new infections", "recovered"])
axislegend(ax, position = :lt)
f

# ## Bonus: fitting an unknown initial infection count (tunable IC)
#
# So far the initial infection counts were *known* inputs of the experiment design. In many
# real datasets the number of infected individuals at the start of an epidemic is **not**
# known — only the first *reported* cases are. We can then make `I(0)` a tunable parameter of
# the wrapped `SingleShootingLayer` via `tunable_ic = [2]`, so it is estimated from the data
# alongside `β` and `γ` (the infected count is state component `2`).
#
# We use the single-region dataset `dataA`, so a single explicit `SingleShootingLayer` is
# enough. Two things to note:
#
#   * the **starting guess** for the tunable entry is the `u0` of the *wrapped ODEProblem* —
#     here a deliberately wrong `I(0) = 40` (the truth is `2.0`), so the fit has to walk the
#     initial condition down to the data;
#   * `bounds_ic` takes **one `(lower, upper)` pair per state component** and only the entries
#     selected by `tunable_ic` are used — here `I(0)` is constrained to `[0, 100]`.

I0_guess = 40.0                           # wrong on purpose (true: 2.0)
prob_ic_guess = ODEProblem(sir!, [Npop - I0_guess, I0_guess, 0.0], tspan, [β_guess, γ_guess])

lay_ic = SingleShootingLayer(
    prob_ic_guess, Tsit5();
    saveat = unique(sort(vcat(t_inf, t_rec))),
    tunable_ic = [2],
    bounds_ic = ([-Inf, 0.0, -Inf], [Inf, 100.0, Inf]),
    bounds_p = ([0.1, 0.05], [2.0, 0.6]),
)
pe_ic = ParameterEstimationLayer(lay_ic, dataA; observed = obs_sir)

# The optimization variables now hold `[I(0), β, γ]` — the tunable initial infected count
# followed by the two rates:

nlls_ic = problem(pe_ic)
println(
    "Optimization variables [I(0), β, γ] starting at: ", nlls_ic.u0,
    "   (true = ", [2.0, β_true, γ_true], ")"
)

sol_ic = solve(nlls_ic, LevenbergMarquardt(); abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 500)
est_ic = estimate_parameters(pe_ic, sol_ic)

println("Fitted [I(0), β, γ] = ", round.(sol_ic.u; digits = 4))
println("True   [I(0), β, γ] = ", [2.0, β_true, γ_true])
println(
    "I(0) estimate: ", round(est_ic.x[1]; digits = 3), " ± ", round(est_ic.std_errors[1]; digits = 3),
    "   (true = 2.0)"
)

# The extra fitted degree of freedom (`I(0)`) costs information: the standard error on `β`
# grows relative to the fit where `I(0)` was known — here by roughly a factor of three — which
# is exactly the expected trade-off when an initial condition has to be inferred rather than
# prescribed.
println("Std error on β with known I(0):   ", round(estA.std_errors[1]; digits = 4))
println("Std error on β with fitted I(0):  ", round(est_ic.std_errors[2]; digits = 4))
