#src ---
#src title: Parameter Estimation for the Lotka-Volterra ODE
#src description: Fit the Lotka-Volterra model parameters (with uncertainties) from a single noisy time-series experiment using a Gauss-Newton least-squares fit
#src tags:
#src   - Lotka Volterra
#src   - Parameter Estimation
#src   - Least Squares
#src icon: 🎯
#src ---

# ## [Plain parameter estimation for the Lotka-Volterra model](@id corleone_pe_lotka_plain)
#
# In this example we estimate the parameters of the classical Lotka-Volterra predator-prey
# ODE from a single noisy time-series experiment. Everything is kept deliberately minimal:
# one experiment, one observation grid, the initial condition prescribed by the design -- the
# only unknowns are the four model parameters. We use a Gauss-Newton least-squares fit and
# then read off the parameter uncertainties (covariance, standard errors and asymptotic
# confidence intervals) from the residual Jacobian at the optimum.

# ## Setup

using Corleone, CorleoneParameterEstimation
using CairoMakie
using OrdinaryDiffEqTsit5
using NonlinearSolve
using StableRNGs

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
# We integrate the true model on a fine grid and add Gaussian noise with a known
# standard deviation `σ`.

rng = StableRNG(42)
tsteps = collect(0.0:0.1:10.0)
M = reduce(hcat, solve(prob, Tsit5(); saveat = tsteps).u)
Y = M .+ 0.05 .* randn(rng, size(M))
data = ExperimentData(tsteps, Y; σ = 0.05)

# The noise scale `σ = 0.05` is stored in the data; the fit uses *weighted* residuals
# `(y_obs - y_model) / σ`, so the uncertainties are computed on the right scale.

# ## Build the parameter-estimation layer
# `ParameterEstimationLayer` wraps one Corleone `SingleShootingLayer` per experiment
# together with the observation map and the data. The layer of each experiment is built
# from the same problem and solved with `saveat = data.t[i]` automatically, so the model is
# sampled exactly at the observation times.

pe = ParameterEstimationLayer(prob, Tsit5(), data)

# By default only the model parameters are treated as tunable degrees of freedom
# (the initial conditions and controls are held fixed). Constructing the
# `NonlinearLeastSquaresProblem` gives a flat vector of optimization variables that starts
# at the initial parameter guess.

nlls_prob = problem(pe)
println("Initial parameter guess: ", nlls_prob.u0)

# ## Solve the least-squares problem
# Both `GaussNewton()` and `LevenbergMarquardt()` are available.

sol = solve(nlls_prob, GaussNewton(); abstol = 1.0e-8, reltol = 1.0e-8)
println("Fitted parameters: ", sol.u)

# ## Estimates and uncertainties
# `estimate_parameters` returns the fitted parameters together with a covariance matrix
# estimated from the residual Jacobian, standard errors and asymptotic 95% confidence
# intervals. Since the data specifies the measurement noise `σ`, the residuals (and hence
# the Jacobian) are the *weighted* ones and the covariance is `(JᵀJ)⁻¹`.

est = estimate_parameters(pe, sol)
println("Standard errors: ", est.std_errors)
println("Confidence intervals: ", est.confidence_intervals)

# The fitted trajectory is returned alongside the estimate through `fitted_parameters`,
# which is convenient for plotting.

result = fitted_parameters(pe, sol)

# With `CairoMakie` loaded, the `CorleoneCairoMakieExtension` registers two recipes that
# plot a fit through the plain Makie `plot!` function:
#
#   * `plot!(ax, result)` overlays the fitted model trajectories (lines) with the
#     observed data (points), one color per observed quantity,
#   * `plot!(ax, data)` plots only the observed data.

f = Figure()
ax = CairoMakie.Axis(f[1, 1], xlabel = "t", ylabel = "population", title = "Lotka-Volterra fit")
plot!(ax, result)      # fitted lines + observed data for both species
axislegend(ax)
display(f)

# ## Fitting an unknown initial condition
#
# If the initial condition of the experiment were *unknown* (rather than a known control of
# the experiment design), it could be made tunable on the wrapped `SingleShootingLayer` via
# `tunable_ic`. The optimization variables then include the tunable initial condition in
# addition to the parameters, which is useful when the initial state itself has to be
# estimated from the data.

pe_ic = ParameterEstimationLayer(
    SingleShootingLayer(prob, Tsit5(); saveat = unique(tsteps), tunable_ic = [1, 2]),
    data,
)
println("Optimization variables when the IC is also tunable: ",
    length(problem(pe_ic).u0), " (4 parameters + 2 initial-state components)")

# ## Solving with an Optimization.jl solver (e.g. UnoSolver)
#
# Besides the primary NonlinearSolve least-squares interface, `CorleoneParameterEstimation` ships a
# thin extension (`CorleoneParameterEstimationOptimizationExtension`) that turns a
# `ParameterEstimationLayer` into an `Optimization.jl` `OptimizationProblem` with the scalar
# least-squares objective ``0.5 \\sum_i r_i^2``. This lets us run the *same* fit with *any*
# Optimization.jl / MathOptInterface solver -- here the state-of-the-art `UnoSolver`
# (FilterSQP). The extension is opt-in and needs `Optimization` and `ComponentArrays` loaded;
# the MOI optimizer is wired in through `OptimizationMOI`.

using Optimization, OptimizationMOI, UnoSolver, ComponentArrays

optprob = OptimizationProblem(pe)                # scalar 0.5 * Σ rᵢ² objective

solver = UnoSolver.Optimizer()
sol_uno = solve(optprob, solver)
println("UnoSolver fitted parameters:     ", sol_uno.u)
println("GaussNewton fitted parameters:   ", sol.u)

# Both solvers reach the same least-squares optimum: UnoSolver minimizes the scalar
# `0.5 * Σ rᵢ²`, while GaussNewton descends on the underlying (weighted) residual vector.
