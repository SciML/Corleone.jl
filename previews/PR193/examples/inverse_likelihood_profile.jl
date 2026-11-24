using CorleoneInverse, OrdinaryDiffEqTsit5, Optimization, OptimizationOptimJL
using LikelihoodProfiler, CairoMakie
using Distributions: Chisq, quantile
using TransformVariables: asℝ₊
using SciMLBase: successful_retcode

t = collect(0.1:0.1:2.0)
σ = 0.05
observed = 1 .+ 0.7 .* t .^ 2 ./ 2 .+ 0.3 .* t .+ 0.01 .* cos.(1:length(t))
experiment = Experiment(1 => (t, observed, σ))
ode = ODEProblem(
    (u, p, t) -> [p[1] * t + p[2]], [1.0], (0.0, 2.0), [0.5, 0.2];
    abstol = 1.0e-10, reltol = 1.0e-10
)
shooting = ShootingLayer(
    ode, Tsit5(), Tunable(1; transform = asℝ₊) => nothing; controls = [2]
)

model = CorleoneDynamicOptProblem(shooting, experiment; reducer = x -> sum(x) / 2)
problem = OptimizationProblem(
    model, Optimization.AutoForwardDiff();
    lb = [log(0.2), -0.5], ub = [log(1.5), 1.0]
)
fit = solve(problem, LBFGS(); abstol = 1.0e-10)
@assert successful_retcode(fit)
estimate = ParameterEstimate(model, fit.u)
@assert maximum(abs.(estimate.x .- [0.7, 0.3])) < 0.05
show(stdout, MIME"text/plain"(), estimate)
println()

cutoff = quantile(Chisq(1), estimate.level)
profile = ProfileLikelihoodProblem(
    problem, estimate;
    profile_lower = log(0.2), profile_upper = log(1.5), threshold = cutoff / 2
)
@assert profile.target.idxs == [1]
@assert profile.optpars ≈ fit.u
method = OptimizationProfiler(;
    optimizer = LBFGS(), stepper = FixedStep(initial_step = 0.025),
    optimizer_opts = (; abstol = 1.0e-10)
)
profiles = solve(profile, method; maxiters = 200)
interval = only(LikelihoodProfiler.endpoints(profiles; xtransform = exp))
@assert !isnothing(interval.left) && !isnothing(interval.right)
@assert interval.left < estimate.x[1] < interval.right
println("Profile interval for a: ", interval)

with_controls = ProfileLikelihoodProblem(
    problem, estimate; include_controls = true,
    profile_lower = [log(0.2), -0.5], profile_upper = [log(1.5), 1.0],
    threshold = cutoff / 2
)
control_only = ProfileLikelihoodProblem(
    problem, estimate; idxs = 2,
    profile_lower = -0.5, profile_upper = 1.0, threshold = cutoff / 2
)
@assert with_controls.target.idxs == [1, 2]
@assert control_only.target.idxs == [2]

curve = only(profiles.profiles)
@assert length(curve.x) > 2
figure = Figure(; size = (900, 360))
data_axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "Observed state")
scatter!(data_axis, t, observed; label = "Observations")
prediction = 1 .+ estimate.x[1] .* t .^ 2 ./ 2 .+ estimate.x[2] .* t
lines!(data_axis, t, prediction; label = "Fitted trajectory")
axislegend(data_axis; position = :lt)
profile_axis = CairoMakie.Axis(
    figure[1, 2]; xlabel = "Rate a", ylabel = "2 Δ negative log-likelihood"
)
lines!(profile_axis, exp.(curve.x), 2 .* (curve.obj .- minimum(curve.obj)); label = "Profile")
hlines!(profile_axis, [cutoff]; color = :red, label = "95% cutoff")
vlines!(profile_axis, collect(estimate.confidence_intervals[1]); linestyle = :dash, color = :gray, label = "Asymptotic interval")
axislegend(profile_axis; position = :ct)
figure

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
