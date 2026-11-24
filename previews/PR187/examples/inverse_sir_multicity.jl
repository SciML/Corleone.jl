using Corleone: SciMLBase
using CorleoneInverse, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5, NonlinearSolve
using StableRNGs, CairoMakie, LinearAlgebra
using TransformVariables: asℝ₊

function sir!(du, u, p, t)
    S, I, R = u
    β, γ = p
    incidence = β * S * I / (S + I + R)
    du[1] = -incidence
    du[2] = incidence - γ * I
    du[3] = γ * I
    return nothing
end
incidence(u, p, t) = p[1] * u[1] * u[2] / sum(u)
rng = StableRNG(3)
N = 1.0e4
p_true = [0.5, 0.2]
prob = ODEProblem(sir!, [N - 2, 2.0, 0.0], (0.0, 30.0), p_true; abstol = 1.0e-8, reltol = 1.0e-8)
t_inf, t_rec = collect(0.0:2.0:30.0), collect(0.0:4.0:30.0)
function measure(u0)
    truth = solve(remake(prob; u0), Tsit5(); saveat = sort(unique(vcat(t_inf, t_rec))))
    infections = [incidence(truth(t), p_true, t) for t in t_inf] .+ 15 .* randn(rng, length(t_inf))
    recovered = [truth(t)[3] for t in t_rec] .+ 150 .* randn(rng, length(t_rec))
    return Experiment(:(p₁ * u₁ * u₂ / (u₁ + u₂ + u₃)) => (t_inf, infections, 15.0), 3 => (t_rec, recovered, 150.0); u0)
end
data_a, data_b = measure([N - 2, 2.0, 0.0]), measure([N - 20, 20.0, 0.0])
shooting = ShootingLayer(prob, Tsit5())
shared = Tunable(1:2; default = [0.3, 0.4], transform = asℝ₊) => nothing
experiments = Experiments(prob, shared; experiments = (region_a = data_a, region_b = data_b))
ps, st = LuxCore.setup(rng, experiments)
model = CorleoneDynamicOptProblem(shooting, experiments; ps, st)
fit = solve(NonlinearLeastSquaresProblem(model), LevenbergMarquardt(); maxiters = 300)
@assert SciMLBase.successful_retcode(fit)
estimate = ParameterEstimate(model, fit.u)
println(estimate)
R0 = estimate.x[1] / estimate.x[2]
gradient = [1 / estimate.x[2], -estimate.x[1] / estimate.x[2]^2]
println("R₀ = ", R0, " ± ", sqrt(gradient' * estimate.covariance * gradient))
@assert all(first(interval) > 0 for interval in estimate.confidence_intervals)

single = Experiments(prob, shared; experiments = (region_a = data_a,))
ps_a, st_a = LuxCore.setup(rng, single)
model_a = CorleoneDynamicOptProblem(shooting, single; ps = ps_a, st = st_a)
fit_a = solve(NonlinearLeastSquaresProblem(model_a), LevenbergMarquardt(); maxiters = 300)
@assert SciMLBase.successful_retcode(fit_a)
estimate_a = ParameterEstimate(model_a, fit_a.u)
println("Single-region standard errors = ", estimate_a.std_errors)
println("Joint standard errors = ", estimate.std_errors)

local_experiment = Experiments(
    prob, shared; experiments = (
        region_a = (
            observations = data_a,
            annotations = (Tunable(2; kind = Val(:variable), default = 10.0, transform = asℝ₊) => [0.0],),
        ),
    )
)
ps_ic, st_ic = LuxCore.setup(rng, local_experiment)
model_ic = CorleoneDynamicOptProblem(shooting, local_experiment; ps = ps_ic, st = st_ic)
fit_ic = solve(NonlinearLeastSquaresProblem(model_ic), LevenbergMarquardt(); maxiters = 500)
@assert SciMLBase.successful_retcode(fit_ic)
println(ParameterEstimate(model_ic, fit_ic.u))

_, reconstruct = pack(default_vectorizer(), ps)
trajectories, _ = shooting(experiments, reconstruct(fit.u), st)
figure = Figure()
for (i, (trajectory, data)) in enumerate(zip(trajectories, (data_a, data_b)))
    axis = CairoMakie.Axis(figure[1, i]; xlabel = "Day", ylabel = "Count", title = "Region $i")
    times = trajectory.t
    fitted_incidence = [incidence(u, estimate.x, t) for (u, t) in zip(trajectory.u, times)]
    lines!(axis, times, fitted_incidence; label = "Infection flux")
    lines!(axis, times, getindex.(trajectory.u, 3); label = "Recovered")
    for observation in data.observations
        scatter!(axis, observation.t, observation.y; markersize = 5)
    end
    axislegend(axis)
end
figure

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
