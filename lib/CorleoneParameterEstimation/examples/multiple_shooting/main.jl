#src ---
#src title: Multiple Shooting Parameter Estimation with an OED-Designed Control
#src description: Estimate the Lotka-Volterra fishing parameters from a multiple-shooting fit under a fixed OED-designed fishing control, showing that a default shooting weight yields a discontinuous solution and a stronger penalty on the continuity conditions is needed
#src tags:
#src   - Lotka Volterra
#src   - Parameter Estimation
#src   - Multiple Shooting
#src   - Optimal Experimental Design
#src   - Fixed Controls
#src icon: 🎯
#src ---

# ## [Multiple-shooting parameter estimation with controls from the OED solve](@id corleone_pe_ms_lotka_oed)
#
# This example combines two ingredients: a **multiple-shooting** layer and the **fixed control**
# produced by an optimal experimental design (OED) solve. The model is the Lotka-Volterra
# fishing model from the [OED-controlled example](@ref corleone_pe_lotka_oed_control): a
# designed, time-varying fishing control is a *known fixed input* of the experiment and the two
# interaction parameters `p[2], p[3]` are the unknowns.
#
# Instead of integrating the whole horizon in one forward pass (single shooting), the time
# interval is partitioned at a *shooting point* and a local state is estimated on each
# sub-interval together with the model parameters. **Continuity ("shooting") residuals** are
# then *appended* to the least-squares problem to keep the reconstructed trajectory continuous
# across the partition.
#
# The central point of this example is a *weighting* issue: the observations are residualized
# against their noise scale `σ` (so each is `O(1)` at the noise floor), but the appended
# shooting residuals are raw state differences and stay on the **state scale** -- here `O(1)`,
# much larger than `σ = 0.02`. With a shooting point at `t = 6` the optimizer can therefore
# stall before it drives the continuity gap to zero, and the *default-weight* fit produces a
# visibly **discontinuous** ODE solution at the shooting point. A *higher penalty* on the
# shooting conditions (`shooting_scale`) is then needed to drive the gap to zero and obtain a
# (numerically) continuous solution.

# ## Setup

using Corleone, CorleoneParameterEstimation
using CorleoneOED
using CairoMakie
using OrdinaryDiffEqTsit5
using NonlinearSolve
using Optimization, OptimizationMOI, Ipopt, ComponentArrays
using LuxCore
using StableRNGs
using LinearAlgebra
using Random

# ## Model dynamics and the OED-designed control
#
# The fishing model: `p[1]` is the (piecewise-constant) fishing control that *replaces* the
# prey growth parameter, and `p[2], p[3]` are the two interaction parameters we want to
# estimate.

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

fish_grid = collect(0.0:0.2:11.8)                      # control grid (piecewise-constant)
measure_times = collect(0.0:0.25:11.75)                # candidate observation times
obs_fish = (u, p, t) -> u[1:2]                         # both species are measurable

# ## a) Solve the optimal experimental design problem
#
# We wrap the model in an [`CorleoneOED.OEDLayer`](@ref) with the tunable parameters
# `params = [2, 3]`, the fishing control on index `1` and two sampling controls (one per
# measured component). Minimising the `DCriterion` yields the most informative fishing
# strategy.

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

# Read off the designed fishing control and the per-species OED-selected sampling times.
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

# ## b) Simulate the designed experiment
#
# We apply the designed fishing control and observe prey and predator **only at their own
# OED-selected sampling times** (`w1` and `w2`, which differ between the two species), adding
# Gaussian noise -- the *per-quantity observation times* form of `Experiment`.

function simulate_species_times(cvals, noise_rng; σ = 0.02)
    c1 = (1 => ControlParameter(fish_grid, controls = cvals, bounds = (0.0, 1.0)),)
    lay = SingleShootingLayer(prob_fish, Tsit5(); controls = c1, saveat = unique(measure_pts))
    ps, st = LuxCore.setup(noise_rng, lay)
    ps = merge(ps, (p = fish_p_true, controls = cvals))
    sol, _ = lay(nothing, ps, st)
    dummy = CorleoneParameterEstimation.ParameterEstimationLayer(
        lay, ExperimentData(measure_pts, zeros(2, length(measure_pts))); observed = obs_fish)
    M = CorleoneParameterEstimation.model_values(dummy, sol, measure_pts)
    prey = M[1, indexin(w1, measure_pts)] .+ σ .* randn(noise_rng, length(w1))
    predator = M[2, indexin(w2, measure_pts)] .+ σ .* randn(noise_rng, length(w2))
    return prey, predator
end

prey_ctrl, pred_ctrl = simulate_species_times(designed_control, rng_f)
data_control = ExperimentData(Experiment([w1, w2], [prey_ctrl, pred_ctrl]; σ = 0.02, u0 = fish_u0,
    controls = (1 => ControlParameter(fish_grid, controls = designed_control, bounds = (0.0, 1.0)),)))

# ## c) Build the multiple-shooting layer
#
# We pass a *single* shooting point `sp = [6.0]`, which partitions the horizon into `(0, 6)`
# and `(6, 12)` and introduces a local state at `t = 6`. The designed fishing control still
# enters in exactly the same way -- a fixed `ControlParameter` in `controls` that is *not* an
# optimization variable.

shooting_point = [6.0]
pe_ms = ParameterEstimationLayer(prob_fish, Tsit5(), data_control, shooting_point; observed = obs_fish)

# The least-squares residual vector has two blocks: the **front block** holds the (weighted)
# *observation* residuals ``(y_{\\text{obs}} - y_{\\text{model}})/\\sigma``, and the **back
# block** holds the *shooting* residuals appended to them -- one continuity residual per
# matching state/parameter/control at each shooting boundary. With the fishing control
# **fixed** these are `2` state (prey, predator) plus `2` parameter (the estimable
# `p[2], p[3]`, since `p[1]` is the control slot) and **0** control matchings -- a fixed
# control is a known input shared by both intervals, not a shooting unknown.

n_obs = n_observations(data_control)
n_shoot = n_shooting_constraints(pe_ms)
println("Observation residuals:      ", n_obs)
println("Appended shooting residuals:", n_shoot)
println("Total residuals:            ", n_obs + n_shoot)

# ## d) Fit with the default shooting weight: a discontinuous solution
#
# The observations are residualized against their noise scale `σ` (so each is `O(1)` at the
# noise floor), but the appended shooting residuals are raw state differences on the **state
# scale** (here `O(1)`, much larger than `σ = 0.02`). With the default (unit) weight the
# optimizer can stall before it drives the continuity gap at `t = 6` to zero. We solve and
# *measure* the shooting residual left in the solution.

sol_ms = solve(problem(pe_ms; x0 = vcat([1.3, 0.8], [1.0, 1.0])), GaussNewton();
    abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 500)
est_ms = estimate_parameters(pe_ms, sol_ms)
println("MS (default weight):  fitted p = ", round.(est_ms.x; digits = 3), "  true = ", fish_p_true)
println("MS (default weight):  std errors = ", round.(est_ms.std_errors; digits = 5))

# A helper to read off the raw (unweighted) shooting residuals from the flat solution vector:
# the last `n_shooting_constraints` entries of the residual vector.
function shooting_residual(pe, x)
    cp = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(pe)
    ps = CorleoneParameterEstimation._unflatten(pe, cp.ps0s, x)
    r = CorleoneParameterEstimation.residual(pe, ps, cp.st)
    return r[(n_observations(pe.data) + 1):end]
end
tic_ms = shooting_residual(pe_ms, sol_ms.u)
println("MS (default weight):  shooting residual = ", round.(tic_ms; digits = 4),
    "  |x| = ", round(norm(tic_ms); digits = 4))

# The shooting residual left at the optimum is on the **state scale** (a visible discontinuity
# at `t = 6`), because the shooting constraints are unweighted while the observations are
# weighted by `1/σ`. This is exactly the situation described in the introduction: the fit
# satisfies the data but leaves a discontinuous ODE solution.

# We visualise this discontinuity by plotting the raw (unweighted) trajectory pieces on each
# sub-interval. The two sub-interval trajectories do **not** meet at `t = 6`:
traj_default = fitted_parameters(pe_ms, sol_ms).trajectory

# ## e) Retry with a higher penalty on the shooting conditions
#
# [`CorleoneParameterEstimationProblem`](@ref) accepts a `shooting_scale` keyword that
# multiplies the appended shooting residuals, letting us weight the continuity constraints
# relative to the data fit. If the optimal solution shows a large shooting residual, we
# increase `shooting_scale` and re-solve -- a standard continuation on the constraint weight
# (here `1000` drives the gap to `~1e-4`, negligible next to the `O(1)` state scale).

sol_ms2 = solve(problem(pe_ms; x0 = vcat([1.3, 0.8], [1.0, 1.0]), shooting_scale = 1000), GaussNewton();
    abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 1000)
est_ms2 = estimate_parameters(pe_ms, sol_ms2)
tic_ms2 = shooting_residual(pe_ms, sol_ms2.u)
println("MS (scaled x1000):   fitted p = ", round.(est_ms2.x; digits = 3))
println("MS (scaled x1000):   shooting residual = ", round.(tic_ms2; digits = 6),
    "  |x| = ", round(norm(tic_ms2); digits = 6))

# Both solves recover the same parameters, but the `shooting_scale = 1000` solve drives the
# continuity residual to (near) zero, so the reconstructed multiple-shooting trajectory is
# (numerically) continuous across the shooting point.

result_ms = fitted_parameters(pe_ms, sol_ms2)
# qualify `Figure`/`Axis` as `ComponentArrays` also exports `Axis`/`axis`
f = CairoMakie.Figure()
ax = CairoMakie.Axis(f[1, 1], xlabel = "t", ylabel = "population", title = "Multiple-shooting fit with OED-designed fishing control")
vlines!(ax, [6.0]; color = (:gray, 0.5), label = "shooting point t = 6")
# The default-weight solution is discontinuous at the shooting point (semi-transparent dashed):
lines!(ax, collect(traj_default.t), getindex.(traj_default.u, 1); color = (:red, 0.4), linestyle = :dash, label = "default weight (discontinuous)")
lines!(ax, collect(traj_default.t), getindex.(traj_default.u, 2); color = (:red, 0.4), linestyle = :dash)
# The scaled fit is continuous (solid); the recipe overlays its fitted lines with the observed data:
#plot!(ax, f_ms, idxs = [1, 2])
#scatter!(ax, w1, prey_ctrl, markersize = 8, label = "observed prey")
#scatter!(ax, w2, pred_ctrl, markersize = 8, label = "observed predator")
plot!(ax, result_ms)
axislegend(ax, position = :lt, nbanks=2)
display(f)
