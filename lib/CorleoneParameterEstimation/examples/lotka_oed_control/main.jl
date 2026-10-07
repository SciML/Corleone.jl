#src ---
#src title: Parameter Estimation with Fixed Controls from Optimal Experimental Design
#src description: Solve an optimal experimental design problem for the Lotka-Volterra fishing model with CorleoneOED, then estimate the parameters with the designed fishing control fixed, showing the uncertainties shrink compared to an uncontrolled design
#src tags:
#src   - Lotka Volterra
#src   - Optimal Experimental Design
#src   - Parameter Estimation
#src   - Fixed Controls
#src icon: 🎯
#src ---

# ## [Parameter estimation with fixed controls from optimal experimental design](@id corleone_pe_lotka_oed_control)
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

# ## Setup

using Corleone, CorleoneParameterEstimation
using CorleoneOED
using CairoMakie
using LuxCore, Random
using OrdinaryDiffEqTsit5
using NonlinearSolve
using Optimization, OptimizationMOI, Ipopt, ComponentArrays
using StableRNGs

# ## Model dynamics and problem

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

# ## Solve the optimal experimental design problem
#
# We wrap the model in an [`CorleoneOED.OEDLayer`](@ref) with the tunable parameters
# `params = [2, 3]`, the fishing control on index `1` and two sampling controls (one per
# measured component). Minimising the `DCriterion` (the determinant of the inverse Fisher
# information matrix) yields the most informative fishing strategy.

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

# ## Parameter estimation with the designed control fixed
#
# Now the same experiment is *actually run*: we apply the designed fishing control, and we
# observe prey and predator **only at their own OED-selected sampling times** (`w1` and `w2`,
# which differ between the two species), adding Gaussian noise. This is expressed with the
# *per-quantity observation times* form of `Experiment` (`Experiment([w1, w2], [prey, predator])`),
# in which each measured quantity is sampled at its own time points instead of a single shared
# grid. The designed control is supplied through `controls = (1 => ControlParameter(...),)` and
# is held **fixed** during the fit -- only `p[2], p[3]` are estimated.

function simulate_species_times(cvals, noise_rng; σ = 0.02)
    c1 = (1 => ControlParameter(fish_grid, controls = cvals, bounds = (0.0, 1.0)),)
    lay = SingleShootingLayer(prob_fish, Tsit5(); controls = c1, saveat = unique(measure_pts))
    ps, st = LuxCore.setup(noise_rng, lay)
    ps = merge(ps, (p = fish_p_true, controls = cvals))
    sol, _ = lay(nothing, ps, st)
    dummy = CorleoneParameterEstimation.ParameterEstimationLayer(
        lay, ExperimentData(measure_pts, zeros(2, length(measure_pts))); observed = obs_fish)
    M = CorleoneParameterEstimation.model_values(dummy, sol, measure_pts) # Prey at w1, predator at w2 (each species measured at its own OED times).
    prey = M[1, indexin(w1, measure_pts)] .+ σ .* randn(noise_rng, length(w1))
    predator = M[2, indexin(w2, measure_pts)] .+ σ .* randn(noise_rng, length(w2))
    return prey, predator
end

prey_ctrl, pred_ctrl = simulate_species_times(designed_control, rng_f)
data_control = ExperimentData(Experiment([w1, w2], [prey_ctrl, pred_ctrl]; σ = 0.02, u0 = fish_u0,
    controls = (1 => ControlParameter(fish_grid, controls = designed_control, bounds = (0.0, 1.0)),)))

pe_control = ParameterEstimationLayer(prob_fish, Tsit5(), data_control; observed = obs_fish)
sol_control = solve(problem(pe_control; x0 = [1.3, 0.8]), GaussNewton();
    abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 500)
est_control = estimate_parameters(pe_control, sol_control)
println("With OED control:      fitted p = ", round.(sol_control.u; digits = 3), "  true = ", fish_p_true)
println("With OED control:      std errors = ", round.(est_control.std_errors; digits = 5))

# ## Comparison with the control switched off
#
# As a baseline, the same two parameters are estimated from a *design without fishing*: the
# fishing control is zero (its default value), so `p[1]` stays at its model default `0.0`.
# We pass an *explicit all-zero* control slot so that both fits estimate exactly the same two
# parameters `p[2], p[3]`, and observe at the *same* per-species OED times for a fair
# comparison -- the only difference is the designed (time-varying) control versus fishing off.

prey_free, pred_free = simulate_species_times(zeros(length(fish_grid)), rng_f)
data_free = ExperimentData(Experiment([w1, w2], [prey_free, pred_free]; σ = 0.02, u0 = fish_u0,
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

# Finally we plot the controlled experiment. The `CorleoneCairoMakieExtension` recipe
# does this in one call -- it reads the per-quantity observation times from the data and
# overlays the fitted model trajectory (lines) with the noisy per-species observations
# (points) at their own times.

result_control = fitted_parameters(pe_control, sol_control)
f = CairoMakie.Figure()
ax = CairoMakie.Axis(f[1, 1], xlabel = "t", ylabel = "population", title = "Fit with OED-designed fishing control")
plot!(ax, result_control)
axislegend(ax, position = :lt)
f