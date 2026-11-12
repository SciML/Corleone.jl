# ---
# title: Parameter estimation with controls from optimal experimental design
# description: Design a fishing schedule and fit interaction rates with the input held fixed.
# category: parameter_estimation
# id: inverse_lotka_oed_control
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# Adapted from Christoph Plate (@chplate)'s example in
# [Corleone.jl PR #168](https://github.com/SciML/Corleone.jl/pull/168).

# Optimal experimental design selects a fishing input and measurement densities.
# The subsequent parameter fit treats the applied input as a known condition.
using Corleone: SciMLBase
using CorleoneInverse, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5, NonlinearSolve
using Optimization, OptimizationMOI, Ipopt, StableRNGs, CairoMakie
using TransformVariables: asℝ₊
include(joinpath(pkgdir(CorleoneInverse), "examples", "lotka_oed_control", "design.jl"))

rng = StableRNG(7)
design = fishing_design(rng)
data_control = fishing_observations(design, design.controls, rng)
data_free = fishing_observations(design, zeros(length(design.controls)), rng)

# ## Fixed local control schedules
# The two experiments share definitions for the unknown rates. Each owns its
# control schedule in Lux state. Their measurement grids are per quantity.
shared = Tunable(2:3; default = [1.3, 0.8], transform = asℝ₊) => nothing
function estimation_experiments(data, controls)
    return Experiments(
        design.problem, shared; controls = [1], experiments = (
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
println("Designed-input standard errors = ", estimate.std_errors)
println("Zero-input standard errors = ", estimate_free.std_errors)
println("Fitted interaction rates = ", estimate.x)

# Compare uncertainties numerically. The OED objective optimizes joint information
# at nominal rates, so individual standard errors depend on the resulting design.
figure = Figure()
axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "Population")
for (i, observation) in enumerate(data_control.observations)
    lines!(axis, trajectory.t, getindex.(trajectory.u, i); label = "Fitted species $i")
    scatter!(axis, observation.t, observation.y; label = "Observed species $i")
end
axislegend(axis)
figure
