# ---
# title: Estimation with controls from experimental design
# description: Design a fishing schedule and fit interaction rates with the input held fixed.
# category: parameter_estimation
# id: inverse_lotka_oed_control
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# Good experiments make parameter estimation easier. Here, optimal experimental
# design (OED) first selects a fishing input and measurement times for the
# [fishing model](@ref lotka_fishing). We then fit the two interaction rates to data
# from that designed experiment, treating the applied input as a known condition,
# and compare against an experiment without fishing.
#
# !!! info "What you will learn"
#     How to give each experiment its own fixed control schedule with `Constant`
#     annotations while sharing the unknown rates across experiments.
using Corleone: SciMLBase
using CorleoneInverse, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5, NonlinearSolve
using Optimization, OptimizationMOI, Ipopt, StableRNGs, CairoMakie
using TransformVariables: asℝ₊

# The model, the OED step, and the synthetic measurements live in a small helper
# file shared with the [multiple-shooting example](@ref inverse_multiple_shooting).
# The fishing model uses `p = [v, β, δ, c₁, c₂]` and keeps the catch coefficients
# `c₁, c₂` as `Constant` annotations.
include(joinpath(pkgdir(CorleoneInverse), "examples", "lotka_oed_control", "design.jl"))

rng = StableRNG(7)
design = fishing_design(rng)
data_control = fishing_observations(design, design.controls, rng)
data_free = fishing_observations(design, zeros(length(design.controls)), rng)
(; designed_controls = round.(design.controls; digits = 3), design.measurement_times)

# ## Fixed local control schedules
# The unknown rates `β, δ` are shared and positive. Each experiment owns its
# control schedule as a `Constant` annotation, and the catch coefficients are
# `Constant` for all experiments. Measurement grids are per quantity.
shared = Tunable(2:3; default = [1.3, 0.8], transform = asℝ₊) => nothing
function estimation_experiments(data, controls)
    return Experiments(
        design.problem, shared, catch_coefficients; controls = [1], experiments = (
            run = (
                observations = data,
                annotations = (Constant(1; default = controls) => design.switches,),
            ),
        )
    )
end
controlled = estimation_experiments(data_control, design.controls)
free = estimation_experiments(data_free, zeros(length(design.controls)))
shooting = ShootingLayer(design.problem, Tsit5())

# Both fits only have the two rates as unknowns.
function fit_experiment(experiments)
    ps, st = LuxCore.setup(rng, experiments)
    model = CorleoneDynamicOptProblem(shooting, experiments; ps, st)
    nlls = NonlinearLeastSquaresProblem(model)
    @assert length(nlls.u0) == 2
    fit = solve(nlls, LevenbergMarquardt(); maxiters = 500)
    @assert SciMLBase.successful_retcode(fit)
    estimate = ParameterEstimate(model, fit.u)
    _, reconstruct = pack(default_vectorizer(), ps)
    trajectories, _ = shooting(experiments, reconstruct(fit.u), st)
    return estimate, only(trajectories)
end
estimate, trajectory = fit_experiment(controlled)
estimate_free, _ = fit_experiment(free)
(;
    fitted_rates = estimate.x,
    designed_input_std_errors = estimate.std_errors,
    zero_input_std_errors = estimate_free.std_errors,
)

# !!! note "Comparing uncertainties"
#     The OED objective optimizes the *joint* information at the nominal rates.
#     Individual standard errors therefore depend on the resulting design and need
#     not all shrink.

figure = Figure()
axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "Population")
for (i, observation) in enumerate(data_control.observations)
    lines!(axis, trajectory.t, getindex.(trajectory.u, i); label = "Fitted species $i")
    scatter!(axis, observation.t, observation.y; label = "Observed species $i")
end
axislegend(axis)
figure

# ## Acknowledgements
#
# This example is adapted from Christoph Plate
# ([@chplate](https://github.com/chplate))'s example in
# [Corleone.jl PR #168](https://github.com/SciML/Corleone.jl/pull/168).
