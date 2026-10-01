#src ---
#src title: Parameter Estimation for the Lotka-Volterra ODE
#src description: Fit Lotka-Volterra model parameters (with uncertainties) from noisy time-series data, including a joint fit over multiple experiments with different initial conditions and a fit with fixed controls from optimal experimental design
#src tags:
#src   - Lotka Volterra
#src   - Parameter Estimation
#src   - Least Squares
#src icon: 🎯
#src ---

# ## [Parameter Estimation for the Lotka-Volterra model](@id corleone_pe_lotka)
#
# In this example we estimate the parameters of a Lotka-Volterra ODE from noisy time-series
# observations with a Gauss-Newton least-squares approach. We then add a *second* experiment
# -- run under different initial conditions -- and perform a *joint* fit over both datasets
# with the same model, showing how the parameter uncertainties shrink when more data is used.

# ## Setup

using Corleone, CorleoneParameterEstimation
using CairoMakie
using LuxCore
using Random
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
traj = result.trajectory

f = Figure()
ax = Axis(f[1, 1], xlabel = "t", ylabel = "population", title = "Lotka-Volterra fit")
plot!(ax, traj, idxs = [1, 2])
scatter!(ax, tsteps, Y[1, :], markersize = 4, label = "observed x")
scatter!(ax, tsteps, Y[2, :], markersize = 4, label = "observed y")
axislegend(ax)
display(f)

# ## Refitting when new data arrives
#
# Parameter estimation is often performed iteratively: a first fit is done on an initial
# experiment, and later *new* data becomes available -- for example a second experiment run
# under *different initial conditions*. We then want to estimate the parameters from the
# *joint* dataset so that all available information is used to constrain the *shared* model
# parameters.

# ### 1. First experiment
# Simulate a first experiment started at `u0_1 = [1.0, 1.0]` and fit it on its own. The
# initial conditions are stored in the experiment data via the `u0` keyword.

u0_1 = [1.0, 1.0]
prob_1 = ODEProblem(lotka_volterra!, u0_1, tspan, p_true)
M1 = reduce(hcat, solve(prob_1, Tsit5(); saveat = tsteps).u)
Y1 = M1 .+ 0.05 .* randn(rng, size(M1))
data_1 = ExperimentData(tsteps, Y1; σ = 0.05, u0 = u0_1)

pe_1 = ParameterEstimationLayer(prob_1, Tsit5(), data_1)
sol_1 = solve(problem(pe_1), LevenbergMarquardt(); abstol = 1.0e-8, reltol = 1.0e-8)
est_1 = estimate_parameters(pe_1, sol_1)
println("Fit from the first experiment only: ", sol_1.u)
println("Standard errors (first experiment): ", est_1.std_errors)

# ### 2. Second experiment with a different initial condition
# A second experiment is now run, starting from a *different* initial condition
# `u0_2 = [0.4, 2.5]`, observed on the same time grid.

u0_2 = [0.4, 2.5]
prob_2 = ODEProblem(lotka_volterra!, u0_2, tspan, p_true)
M2 = reduce(hcat, solve(prob_2, Tsit5(); saveat = tsteps).u)
Y2 = M2 .+ 0.05 .* randn(rng, size(M2))
data_2 = ExperimentData(tsteps, Y2; σ = 0.05, u0 = u0_2)

# ### 3. Joint fit over the combined data
#
# The parameters are shared between the experiments, but each experiment starts from its
# own initial conditions. An `ExperimentData` can hold *multiple* experiments: both are
# simulated with the **same** model -- the shooting layer of each experiment is built from
# the same problem with that experiment's initial conditions and observation times -- and
# the residuals of all experiments are concatenated into a single least-squares problem.

data_joint = ExperimentData(data_1, data_2)
println("Number of experiments: ", n_experiments(data_joint))
println("Total number of observations: ", n_observations(data_joint))

pe_joint = ParameterEstimationLayer(prob_1, Tsit5(), data_joint)

# To highlight the *incremental* nature of the refit, we warm-start the joint fit from the
# result of the first fit instead of the initial guess. The flat optimization variables of
# `pe_joint` contain the tunable *parameters only* -- here the four model parameters shared
# by both experiments, since the initial conditions are prescribed by the data. A warm
# start is therefore simply `sol_1.u`:

joint_prob = problem(pe_joint; x0 = sol_1.u)
sol_joint = solve(joint_prob, LevenbergMarquardt(); abstol = 1.0e-8, reltol = 1.0e-8)
est_joint = estimate_parameters(pe_joint, sol_joint)

println("First-experiment fit:   ", sol_1.u)
println("Joint fit:              ", sol_joint.u)
println("True parameters:        ", p_true)

# The joint fit uses both experiments, so the parameter uncertainties shrink relative to
# the first-experiment-only fit:
println("Standard errors (first experiment): ", est_1.std_errors)
println("Standard errors (joint fit):        ", est_joint.std_errors)

# The joint fit produces one fitted trajectory per experiment, which we can plot together
# with the observations of both runs.

result_joint = fitted_parameters(pe_joint, sol_joint)
traj_1, traj_2 = result_joint.trajectory

f2 = Figure()
ax2 = Axis(f2[1, 1], xlabel = "t", ylabel = "population", title = "Joint fit over two experiments")
plot!(ax2, traj_1, idxs = [1, 2])
plot!(ax2, traj_2, idxs = [1, 2])
scatter!(ax2, tsteps, Y1[1, :], markersize = 4, label = "exp 1: observed x")
scatter!(ax2, tsteps, Y1[2, :], markersize = 4, label = "exp 1: observed y")
scatter!(ax2, tsteps, Y2[1, :], markersize = 4, marker = :diamond, label = "exp 2: observed x")
scatter!(ax2, tsteps, Y2[2, :], markersize = 4, marker = :diamond, label = "exp 2: observed y")
axislegend(ax2, position = :rt)
display(f2)

# ### Fitting an unknown initial condition
#
# If the initial condition of an experiment were *unknown* (rather than a known control of
# the experiment design), it could be made tunable on the wrapped `SingleShootingLayer` via
# `tunable_ic`. The optimization variables then include the tunable initial condition in
# addition to the parameters, which is useful when the initial state itself has to be
# estimated from the data. For the single-experiment setup above:

pe_ic = ParameterEstimationLayer(
    SingleShootingLayer(prob, Tsit5(); saveat = unique(tsteps), tunable_ic = [1, 2]),
    data,
)

# ## Fitting parameters of an [optimal experimental design](@id corleone_pe_oed)
#
# Optimal experimental design (OED) solves for a *time-varying control* (here a fishing
# rate) that maximises the information content of the data: the experiment is most
# informative exactly where the parameters are most sensitive. After such a designed
# experiment has been *carried out in the real world*, we collect measurements and
# estimate the model parameters. The designed control is then a **known, fixed input** of the
# experiment -- it is *not* an optimization variable. [`CorleoneOED`](https://github.com/SciML/Corleone.jl)
# produces that control, [`CorleoneParameterEstimation`](@ref) fits the parameters with it.
#
# We use the canonical Lotka-Volterra fishing model: `p[1]` is the (piecewise-constant)
# fishing control that *replaces* the prey growth parameter, and `p[2], p[3]` are the two
# interaction parameters we want to estimate.

# ### a) Model dynamics and problem

function fishing_lotka!(du, u, p, t)
    du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    return
end
fish_tspan = (0.0, 12.0)
fish_u0 = [0.5, 0.7]
fish_p0 = [0.0, 1.0, 1.0]        # p[1] default = 0 (no fishing) unless the control acts
fish_p_true = [1.0, 1.0]         # the two tunable parameters (p[2], p[3])
prob_fish = ODEProblem(fishing_lotka!, fish_u0, fish_tspan, fish_p0)
rng_f = StableRNG(7)

# The fishing control and the candidate measurement times share a common discretization.

fish_grid = collect(0.0:0.2:11.8)                      # control grid (piecewise-constant)
measure_times = collect(0.0:0.25:11.75)                # candidate observation times
obs_fish = (u, p, t) -> u[1:2]                         # both species are measurable

# ### b) Solve the optimal experimental design problem
#
# We wrap the model in an [`CorleoneOED.OEDLayer`](@ref) with the tunable parameters
# `params = [2, 3]`, the fishing control on index `1` and two sampling controls (one per
# measured component). Minimising the `DCriterion` (the determinant of the inverse Fisher
# information matrix) yields the most informative fishing strategy.

using CorleoneOED
using Optimization, OptimizationMOI, Ipopt, ComponentArrays

fishing_control = ControlParameter(
    fish_grid, name = :fishing, bounds = (0.0, 1.0), controls = ones(length(fish_grid)))
oed = OEDLayer{false}(
    prob_fish, Tsit5();
    params = [2, 3],
    controls = (1 => fishing_control,),
    bounds_p = ([1.0, 1.0], [2.0, 2.0]),
    observed = obs_fish,
    measurements = [
        ControlParameter(measure_times, controls = 0.5 * ones(length(measure_times)), bounds = (0.0, 1.0)),
        ControlParameter(measure_times, controls = 0.5 * ones(length(measure_times)), bounds = (0.0, 1.0)),
    ],
)

optprob = OptimizationProblem(oed, DCriterion(); M = [4.0, 4.0])
uopt = solve(optprob, Ipopt.Optimizer(); tol = 1.0e-8,
    hessian_approximation = "limited-memory", max_iter = 300)
println("OED solve successful: ", SciMLBase.successful_retcode(uopt))

# The optimal design stores the fishing control values first in its `controls` block, so
# we can read off the designed control for the experiment. The two sampling controls `w₁`
# and `w₂` (one per measured species) tell us *when* to measure: they are active (≈ 1) at
# the optimized observation points. We collect the **union** of the times chosen for either
# species and observe both species at those times (per-quantity observation times are a
# natural extension, since each experiment's observation schedule is stored separately).

ps_oed, _ = LuxCore.setup(Random.default_rng(), oed)
cc = uopt.u + zero(ComponentArray(ps_oed))
nfish = length(fish_grid)
nm = length(measure_times)
designed_control = collect(cc.controls[1:nfish])
w1 = measure_times[collect(cc.controls[nfish .+ (1:nm)]) .> 0.5]
w2 = measure_times[collect(cc.controls[nfish .+ nm .+ (1:nm)]) .> 0.5]
measure_pts = sort(unique(vcat(w1, w2)))
println("Fishing control starts at:   ", round.(designed_control[1:5]; digits = 2), " ...")
println("Species-1 measurement times: ", round.(w1; digits = 2))
println("Species-2 measurement times: ", round.(w2; digits = 2))
println("Union observation times (n = ", length(measure_pts), "): ", round.(measure_pts; digits = 2))

# ### c) Parameter estimation with the designed control fixed
#
# Now the same experiment is *actually run*: we apply the designed fishing control and
# observe both species **only at the times chosen by the OED solution** (the union `measure_pts`),
# adding Gaussian noise. The designed control is supplied to the experiment through
# `controls = (1 => ControlParameter(...),)` and is held **fixed** during the fit -- only
# `p[2], p[3]` are estimated.

function simulate_with_control(cvals, noise_rng, times; σ = 0.02)
    c1 = (1 => ControlParameter(fish_grid, controls = cvals, bounds = (0.0, 1.0)),)
    lay = SingleShootingLayer(prob_fish, Tsit5(); controls = c1, saveat = unique(times))
    ps, st = LuxCore.setup(noise_rng, lay)
    ps = merge(ps, (p = fish_p_true, controls = cvals))
    sol, _ = lay(nothing, ps, st)
    # use the observed map to pull the two species out of the trajectory at `times`
    dummy = CorleoneParameterEstimation.ParameterEstimationLayer(
        lay, ExperimentData(times, zeros(2, length(times))); observed = obs_fish)
    M = CorleoneParameterEstimation.model_values(dummy, sol, times)
    return M .+ σ .* randn(noise_rng, size(M))
end

Y_control = simulate_with_control(designed_control, rng_f, measure_pts)
data_control = ExperimentData(Experiment(measure_pts, Y_control; σ = 0.02, u0 = fish_u0,
    controls = (1 => ControlParameter(fish_grid, controls = designed_control, bounds = (0.0, 1.0)),)))

pe_control = ParameterEstimationLayer(prob_fish, Tsit5(), data_control; observed = obs_fish)
sol_control = solve(problem(pe_control; x0 = [1.3, 0.8]), GaussNewton();
    abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 500)
est_control = estimate_parameters(pe_control, sol_control)
println("With OED control:      fitted p = ", round.(sol_control.u; digits = 3), "  true = ", fish_p_true)
println("With OED control:      std errors = ", round.(est_control.std_errors; digits = 5))

# ### d) Comparison with the control switched off
#
# As a baseline, the same parameters are estimated from a *design without any fishing*: the
# fishing control is simply zero (its default value, i.e. no fishing), so `p[1]` stays at
# its model default `0.0`. Here we pass an *explicit all-zero* control slot so that both
# fits estimate exactly the same two parameters `p[2], p[3]` -- the difference is purely the
# designed (time-varying) control versus the control set to zero. To make the comparison
# fair, we observe at the *same* (union) times chosen by OED.

Y_free = simulate_with_control(zeros(length(fish_grid)), rng_f, measure_pts)
data_free = ExperimentData(Experiment(measure_pts, Y_free; σ = 0.02, u0 = fish_u0,
    controls = (1 => ControlParameter(fish_grid, controls = zeros(length(fish_grid)), bounds = (0.0, 1.0)),)))

pe_free = ParameterEstimationLayer(prob_fish, Tsit5(), data_free; observed = obs_fish)
sol_free = solve(problem(pe_free; x0 = [1.3, 0.8]), GaussNewton();
    abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 500)
est_free = estimate_parameters(pe_free, sol_free)
println("Control off (default):  fitted p = ", round.(sol_free.u; digits = 3))
println("Control off (default):  std errors = ", round.(est_free.std_errors; digits = 5))

# The designed experiment squeezes more information out of the data: the uncertainties on
# the two parameters are smaller than in the uncontrolled design (a smaller standard error
# means the parameter is better constrained).

println("---")
println("OED-derived experiment reduces the p[2] std error by ",
    round(100 * (1 - est_control.std_errors[1] / est_free.std_errors[1]); digits = 1), "%")
println("and the p[3] std error by ",
    round(100 * (1 - est_control.std_errors[2] / est_free.std_errors[2]); digits = 1), "%")

# Finally we plot the controlled experiment: the model trajectory under the designed
# fishing control together with the noisy observations.

f_result = fitted_parameters(pe_control, sol_control).trajectory
# qualify `Figure`/`Axis` as `ComponentArrays` also exports `Axis`/`axis`
f3 = CairoMakie.Figure()
ax3 = CairoMakie.Axis(f3[1, 1], xlabel = "t", ylabel = "population", title = "Fit with OED-designed fishing control")
plot!(ax3, f_result, idxs = [1, 2])
scatter!(ax3, measure_pts, Y_control[1, :], markersize = 4, label = "observed prey")
scatter!(ax3, measure_pts, Y_control[2, :], markersize = 4, label = "observed predator")
axislegend(ax3, position = :rt)
display(f3)

