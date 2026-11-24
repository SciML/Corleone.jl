using Corleone: SciMLBase
using CorleoneInverse, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5, NonlinearSolve
using StableRNGs, CairoMakie
using Optimization, OptimizationMOI, Ipopt
using TransformVariables: asℝ₊

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
    return nothing
end

rng = StableRNG(42)
p_true = [1.5, 1.0, 3.0, 1.0]
prob = ODEProblem(
    lotka_volterra!, [1.0, 1.0], (0.0, 10.0), p_true; abstol = 1.0e-9, reltol = 1.0e-9
)
t = collect(0.0:0.1:10.0)
truth = solve(prob, Tsit5(); saveat = t)
y = reduce(hcat, truth.u) .+ 0.05 .* randn(rng, 2, length(t))

observations = Experiment(1 => (t, y[1, :], 0.05), 2 => (t, y[2, :], 0.05))

layer = ShootingLayer(
    remake(prob; p = p_true .+ 0.05 .* randn(rng, 4)), Tsit5(),
    Tunable(1:4; transform = asℝ₊) => nothing
)
ps, st = LuxCore.setup(rng, layer)
model = CorleoneDynamicOptProblem(layer, observations; ps, st)
nlls = NonlinearLeastSquaresProblem(model)
fit = solve(nlls, LevenbergMarquardt(); abstol = 1.0e-8, reltol = 1.0e-8)
@assert SciMLBase.successful_retcode(fit)
estimate = ParameterEstimate(model, fit.u)
@assert maximum(abs.(estimate.x ./ p_true .- 1)) < 0.15
estimate

_, reconstruct = pack(default_vectorizer(), ps)
trajectory, _ = layer(
    observations, reconstruct(fit.u), experiment_state(layer, observations, st)
)
figure = Figure()
axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "Population")
for i in 1:2
    lines!(axis, trajectory.t, getindex.(trajectory.u, i); label = "Fitted species $i")
    scatter!(axis, t, y[i, :]; markersize = 4, label = "Observed species $i")
end
axislegend(axis)
figure

ic_layer = ShootingLayer(
    prob, Tsit5(), Tunable(1:4; transform = asℝ₊) => nothing,
    Tunable(1:2; kind = Val(:variable), transform = asℝ₊) => [0.0]
)
ic_ps, ic_st = LuxCore.setup(rng, ic_layer)
ic_model = CorleoneDynamicOptProblem(ic_layer, observations; ps = ic_ps, st = ic_st)
ic_fit = solve(NonlinearLeastSquaresProblem(ic_model), LevenbergMarquardt())
@assert length(ic_fit.u) == 6
ParameterEstimate(ic_model, ic_fit.u)

opt = OptimizationProblem(model, Optimization.AutoForwardDiff())
ipopt_fit = solve(opt, Ipopt.Optimizer(); print_level = 0, tol = 1.0e-8)
@assert SciMLBase.successful_retcode(ipopt_fit)
ParameterEstimate(model, ipopt_fit.u)

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
