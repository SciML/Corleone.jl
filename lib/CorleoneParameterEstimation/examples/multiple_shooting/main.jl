#src ---
#src title: Multiple Shooting Parameter Estimation for the Lotka-Volterra Model
#src description: Estimate Lotka-Volterra parameters from noisy time-series data with a Multiple Shooting layer and a single shooting point, showing how the appended continuity residuals partition the horizon into short, tame sub-intervals
#src tags:
#src   - Lotka Volterra
#src   - Parameter Estimation
#src   - Multiple Shooting
#src   - Least Squares
#src icon: 🎯
#src ---

# ## [Multiple Shooting Parameter Estimation for the Lotka-Volterra model](@id corleone_pe_ms_lotka)
#
# This example estimates the parameters of the classical Lotka-Volterra predator-prey model
# from noisy time-series observations using **multiple shooting**. Instead of integrating the
# whole horizon in one forward pass (single shooting), the time interval is partitioned at a
# *shooting point* and a local state is estimated on each sub-interval together with the model
# parameters. **Continuity ("shooting") residuals** are then *appended* to the least-squares
# problem to keep the reconstructed trajectory continuous across the partition.
#
# `CorleoneParameterEstimation` builds a `MultipleShootingLayer` (one per experiment) when you
# pass a vector of shooting points to the layer constructor. The layer automatically appends the
# problem's time span to those points, so passing a single shooting point `[6.0]` over the span
# `(0, 10)` produces the two intervals `(0, 6)` and `(6, 10)`.

# ## Setup

using Corleone, CorleoneParameterEstimation
using CairoMakie
using OrdinaryDiffEqTsit5
using NonlinearSolve
using StableRNGs
using LinearAlgebra

# ## Lotka-Volterra dynamics

# The classical two-species predator-prey model
# ```math
# \begin{aligned}
# \frac{dx}{dt} &= \alpha x - \beta x y,\\
# \frac{dy}{dt} &= -\delta y + \gamma x y,
# \end{aligned}
# ```
# with four parameters `p = (α, β, δ, γ)`.

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
    return
end

tspan = (0.0, 10.0)
u0 = [1.0, 1.0]
p_true = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(lotka_volterra!, u0, tspan, p_true)

# ## Generate noisy observations
# We integrate the true model on the observation grid and add Gaussian noise with a known
# standard deviation `σ`.

rng = StableRNG(42)
tsteps = collect(0.0:0.1:10.0)
M = reduce(hcat, solve(prob, Tsit5(); saveat = tsteps).u)
Y = M .+ 0.05 .* randn(rng, size(M))
data = ExperimentData(tsteps, Y; σ = 0.05)

# The noise scale `σ = 0.05` is stored in the data; the fit uses *weighted* residuals
# `(y_obs - y_model) / σ`, so the uncertainties are computed on the right scale.

# ## Build the multiple-shooting layer
# We pass a *single* shooting point `sp = [6.0]`. The layer appends the problem time span
# automatically, so the shooting intervals are `(0, 6)` and `(6, 10)`. A local state is
# introduced at the second interval's start (`t = 6`), giving two coupled sub-problems.

sp = [6.0]
pe = ParameterEstimationLayer(prob, Tsit5(), data, sp)

# ### The residual vector: observations first, shooting residuals appended
#
# The least-squares residual vector `r(x)` has two blocks. The **front block** holds the
# (weighted) *observation* residuals ``(y_{\\text{obs}} - y_{\\text{model}})/\\sigma`` at every
# measurement time. The **back block** holds the *shooting* residuals, which are **appended**
# to the observation residuals -- one continuity residual per matching state/parameter/control
# at each shooting boundary. They are zero when every interval starts exactly where the
# previous one ended, i.e. when the reconstructed trajectory is continuous.
#
# `n_observations(data)` counts the front block and `n_shooting_constraints(pe)` counts the
# appended continuity residuals, so the total is `n_observations + n_shooting_constraints`.

n_obs = n_observations(data)
n_shoot = n_shooting_constraints(pe)
println("Observation residuals:      ", n_obs)
println("Appended shooting residuals:", n_shoot)
println("Total residuals:            ", n_obs + n_shoot)

# For this problem there are two state components and four shared parameters along one
# shooting boundary, so the six shooting residuals are the two *state* continuity residuals
# (which can be violated when the intervals are inconsistent) followed by the four *parameter*
# continuity residuals (always zero, because both intervals share the same global parameters).

# ## Inspect the initial guess
# The initial conditions are prescribed (`u0 = [1, 1]`), so the optimization variables are the
# four shared parameters plus the two local shooting-state components at `t = 6`. We start from
# a deliberately perturbed parameter guess so that the two sub-intervals do **not** match at the
# shooting point and the shooting residuals are visibly non-zero.

x0 = vcat([1.2, 1.3, 2.8, 1.1], [1.0, 1.0])     # perturbed params + local states
println("Initial parameter guess (shared):  ", x0[1:4])

# Internal state needed to run the layer at an arbitrary flat `x` and read off its
# residuals (see `CorleoneParameterEstimationProblem`).
cp = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(pe)

# The perturbed layer run produces a trajectory that is **discontinuous** at the shooting point:
ps_init = CorleoneParameterEstimation._unflatten(pe, cp.ps0s, x0)
traj_init, _ = pe(nothing, ps_init, cp.st)

# We evaluate the full residual vector at this initial guess and split it into its two blocks.
res_init = residual(pe, ps_init, cp.st)
obs_res_init = res_init[1:n_obs]
shoot_res_init = res_init[(n_obs + 1):end]
println("Residual norm (observations) at initial guess: ", round(norm(obs_res_init); digits = 3))
println("Shooting residuals at initial guess:            ", round.(shoot_res_init; digits = 3))

# The two non-zero entries are the *state* continuity residuals at the shooting point `t = 6`:
# the sub-intervals are inconsistent and must be corrected by the optimizer. The four parameter
# residuals stay zero because the four model parameters are shared by both intervals.

# ## Solve the least-squares problem
# Gauss-Newton descends on the *concatenated* residual vector (observations + shooting), driving
# both the fit error and the continuity violation to zero.

nlls_prob = problem(pe; x0 = x0)
sol = solve(nlls_prob, GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8)
println("Fitted parameters: ", round.(sol.u[1:4]; digits = 4))

# ## Estimates and uncertainties
# `estimate_parameters` reports the shared model block (the four parameters) together with a
# covariance matrix, standard errors and asymptotic 95% confidence intervals. The local
# shooting states are profile-eliminated nuisance variables and are not reported.

est = estimate_parameters(pe, sol)
println("Estimated parameters: ", round.(est.x; digits = 4))
println("True parameters:      ", p_true)
println("Standard errors:      ", round.(est.std_errors; digits = 4))

# After the fit the appended shooting residuals vanish (up to solver tolerance), confirming
# that the continuity constraints are satisfied:
ps_fit = CorleoneParameterEstimation._unflatten(pe, cp.ps0s, sol.u)
res_fit = residual(pe, ps_fit, cp.st)
println("Residual norm (observations) at the fit:        ", round(norm(res_fit[1:n_obs]); digits = 3))
println("Shooting residuals at the fit:                  ", round.(res_fit[(n_obs + 1):end]; digits = 3))

# ## Plot: data, initial guess, and fitted trajectory
# We overlay the noisy observations, the (discontinuous) interrupted initial-guess trajectories
# and the smooth fitted trajectory, marking the shooting point `t = 6` where the continuity
# residuals are enforced.

result = fitted_parameters(pe, sol)
traj = result.trajectory

f = Figure()
ax = Axis(f[1, 1], xlabel = "t", ylabel = "population",
    title = "Multiple-shooting fit of the Lotka-Volterra model")
vlines!(ax, [6.0]; color = (:gray, 0.5), label = "shooting point t = 6")
# Initial (perturbed) guess: one dashed, semi-transparent curve per component.
lines!(ax, collect(traj_init.t), getindex.(traj_init.u, 1); color = (:grey, 0.6), linestyle = :dash)
lines!(ax, collect(traj_init.t), getindex.(traj_init.u, 2); color = (:grey, 0.6), linestyle = :dash, label = "initial guess")
plot!(ax, traj, idxs = [1, 2])
scatter!(ax, tsteps, Y[1, :], markersize = 4, label = "observed prey")
scatter!(ax, tsteps, Y[2, :], markersize = 4, label = "observed predator")
axislegend(ax, position = :rt)
display(f)
CairoMakie.save(joinpath(@__DIR__, "lotka_multiple_shooting.png"), f)
println("\nPlot saved to ", joinpath(@__DIR__, "lotka_multiple_shooting.png"))
