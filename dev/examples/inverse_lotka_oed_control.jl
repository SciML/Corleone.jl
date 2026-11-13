using Corleone: SciMLBase
using CorleoneInverse, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5, NonlinearSolve
using Optimization, OptimizationMOI, Ipopt, StableRNGs, CairoMakie
using TransformVariables: asℝ₊
include(joinpath(pkgdir(CorleoneInverse), "examples", "lotka_oed_control", "design.jl"))

rng = StableRNG(7)
design = fishing_design(rng)
data_control = fishing_observations(design, design.controls, rng)
data_free = fishing_observations(design, zeros(length(design.controls)), rng)

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

figure = Figure()
axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "Population")
for (i, observation) in enumerate(data_control.observations)
    lines!(axis, trajectory.t, getindex.(trajectory.u, i); label = "Fitted species $i")
    scatter!(axis, observation.t, observation.y; label = "Observed species $i")
end
axislegend(axis)
figure

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
