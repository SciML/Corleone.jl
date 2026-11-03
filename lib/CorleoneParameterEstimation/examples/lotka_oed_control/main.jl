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
# produces that control, `CorleoneParameterEstimation` fits the parameters with it.
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
using SymbolicIndexingInterface: parameter_values_at_time, parameter_index

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

fish_grid = collect(0.0:0.25:11.75)                    # control grid (piecewise-constant)
measure_times = collect(0.0:0.25:11.75)                # candidate observation times
obs_fish = (u, p, t) -> u[1:2]                         # both species are measurable

# ## Solve the optimal experimental design problem
#
# We wrap the model in a `CorleoneOED.OEDLayer` with the tunable parameters
# `params = [2, 3]`, the fishing control on index `1` and two sampling controls (one per
# measured component). Minimising the `DCriterion` (the determinant of the inverse Fisher
# information matrix) yields the most informative fishing strategy.

control = Tunable(1; default = 0.0, lowerbound = 0.0, upperbound = 1.0) => fish_grid[2:end]
measurements = (
    ContinuousMeasurement(:w1, measure_times, (x, p, t) -> x[1]),
    ContinuousMeasurement(:w2, measure_times, (x, p, t) -> x[2]),
)
oed = OEDLayer(prob_fish, Tsit5(), control; params = [2, 3], measurements, controls = [1])
ps_oed, st_oed = LuxCore.setup(Random.default_rng(), oed)

model = CorleoneDynamicOptProblem(oed, DCriterion(); ps = ps_oed, st = st_oed, M = [4.0, 4.0])
optprob = OptimizationProblem(model, AutoForwardDiff())

uopt = solve(
    optprob, Ipopt.Optimizer(); tol = 1.0e-8,
    hessian_approximation = "limited-memory", max_iter = 300
)
println("OED solve successful: ", SciMLBase.successful_retcode(uopt))

# Reconstruct the optimal parameter tree and read the designed schedules back with
# symbolic parameter queries: the fishing control at each interval midpoint and the
# measurement densities at the candidate observation times. The two sampling densities
# `w1` and `w2` (one per measured species) tell us *when* to measure: they are active
# (≈ 1) at the optimized observation points. We collect the **union** of the times chosen
# for either species and observe both species at those times (per-quantity observation
# times are a natural extension, since each experiment's observation schedule is stored
# separately).

_, reconstruct = pack(default_vectorizer(), ps_oed)
optimal_ps = reconstruct(uopt.u)
traj_oed, _ = oed(nothing, optimal_ps, st_oed)
edges = vcat(first(prob_fish.tspan), fish_grid[2:end], last(prob_fish.tspan))
designed_control = optimal_ps.parameter.parameters.p₁.parameter.p₁.ps
w1 = optimal_ps.parameter.parameters.p₁.parameter.w1.ps
w2 = optimal_ps.parameter.parameters.p₁.parameter.w2.ps

w1_t = [t for (i, t) in enumerate(measure_times) if w1[i] > 0.5]
w2_t = [t for (i, t) in enumerate(measure_times) if w2[i] > 0.5]
measure_pts = sort(unique(vcat(w1_t, w2_t)))
println("Fishing control starts at:   ", round.(designed_control[1:5]; digits = 2), " ...")
println("Species-1 measurement times: ", round.(w1_t; digits = 2))
println("Species-2 measurement times: ", round.(w2_t; digits = 2))
println("Union observation times (n = ", length(measure_pts), "): ", round.(measure_pts; digits = 2))

# ## Parameter estimation with the designed control fixed
#
# Now the same experiment is *actually run*: we apply the designed fishing control, and we
# observe prey and predator **only at their own OED-selected sampling times** (`w1_t` and
# `w2_t`, which differ between the two species), adding Gaussian noise. This is expressed with
# the *per-quantity observation times* form of `Experiment` (`Experiment([w1_t, w2_t], [prey, predator])`),
# in which each measured quantity is sampled at its own time points instead of a single shared
# grid. The designed control is supplied through
# `controls = (1 => annotation => switching_times,)` and is held **fixed** during the fit --
# only `p[2], p[3]` are estimated.

function simulate_species_times(cvals, noise_rng; σ = 0.02)
    control = Tunable(1; default = cvals, lowerbound = 0.0, upperbound = 1.0) => fish_grid[2:end]
    sim_data = ExperimentData(
        Experiment(
            [w1_t, w2_t], [zeros(length(w1_t)), zeros(length(w2_t))]; σ = 0.02, u0 = fish_u0,
            controls = (1 => control,)
        )
    )
    sim = CorleoneParameterEstimation.ParameterEstimationLayer(
        prob_fish, Tsit5(), sim_data; observed = obs_fish
    )
    ps, st = LuxCore.setup(noise_rng, sim)
    sol, _ = sim(nothing, ps, st)
    M = CorleoneParameterEstimation.model_values(sim, sol, measure_pts) # Prey at w1_t, predator at w2_t (each species measured at its own OED times).
    prey = M[1, indexin(w1_t, measure_pts)] .+ σ .* randn(noise_rng, length(w1_t))
    predator = M[2, indexin(w2_t, measure_pts)] .+ σ .* randn(noise_rng, length(w2_t))
    return prey, predator
end

prey_ctrl, pred_ctrl = simulate_species_times(designed_control, rng_f)
data_control = ExperimentData(
    Experiment(
        [w1_t, w2_t], [prey_ctrl, pred_ctrl]; σ = 0.02, u0 = fish_u0,
        controls = (
            1 => Tunable(1; default = designed_control, lowerbound = 0.0, upperbound = 1.0) =>
                fish_grid[2:end],
        )
    )
)

pe_control = ParameterEstimationLayer(prob_fish, Tsit5(), data_control; observed = obs_fish)
sol_control = solve(
    problem(pe_control; x0 = [1.3, 0.8]), GaussNewton();
    abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 500
)
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
data_free = ExperimentData(
    Experiment(
        [w1_t, w2_t], [prey_free, pred_free]; σ = 0.02, u0 = fish_u0,
        controls = (
            1 => Tunable(1; default = zeros(length(fish_grid)), lowerbound = 0.0, upperbound = 1.0) =>
                fish_grid[2:end],
        )
    )
)

pe_free = ParameterEstimationLayer(prob_fish, Tsit5(), data_free; observed = obs_fish)
sol_free = solve(
    problem(pe_free; x0 = [1.3, 0.8]), GaussNewton();
    abstol = 1.0e-9, reltol = 1.0e-9, maxiters = 500
)
est_free = estimate_parameters(pe_free, sol_free)
println("Control off (default):  fitted p = ", round.(sol_free.u; digits = 3))
println("Control off (default):  std errors = ", round.(est_free.std_errors; digits = 5))

# The designed experiment squeezes more information out of the data: the uncertainties on
# the two parameters are smaller than in the uncontrolled design (a smaller standard error
# means the parameter is better constrained).

println("---")
println(
    "OED-derived experiment reduces the p[2] std error by ",
    round(100 * (1 - est_control.std_errors[1] / est_free.std_errors[1]); digits = 1), "%"
)
println(
    "and the p[3] std error by ",
    round(100 * (1 - est_control.std_errors[2] / est_free.std_errors[2]); digits = 1), "%"
)

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
