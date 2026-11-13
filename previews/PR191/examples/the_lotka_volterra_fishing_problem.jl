using Corleone

using LuxCore
using Random
using OrdinaryDiffEqTsit5
using SymbolicIndexingInterface
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using ForwardDiff
using CairoMakie
Makie.inline!()

function lotka_dynamics(du, u, p, t)
    β, v, δ, c₁, c₂ = p
    du[1] = u[1] - β * u[1] * u[2] - c₁ * v * u[1]
    du[2] = -u[2] + δ * u[1] * u[2] - c₂ * v * u[2]
    du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
    return nothing
end
tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [1.0, 0.0, 1.0, 0.4, 0.2]
prob = ODEProblem(lotka_dynamics, u0, tspan, p0; abstol = 1.0e-9, reltol = 1.0e-9)

cgrid = collect(0.1:0.1:11.9)
control = Tunable(
    2; kind = Val(:parameter), default = 1.0,
    lowerbound = 0.0, upperbound = 1.0
) => cgrid
model_constants = (
    Constant([1, 3, 4, 5]; kind = Val(:parameter)) => nothing,
)
layer = ShootingLayer(
    prob, Tsit5(), control, model_constants...;
    controls = [2], quadratures = [3]
)

dynprob = CorleoneDynamicOptProblem(layer, :(u₃(12.0)))
opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
uopt = solve(
    opt, Ipopt.Optimizer();
    tol = 1.0e-6, hessian_approximation = "limited-memory",
    max_iter = 300, print_level = 0
)
@assert successful_retcode(uopt)
(; retcode = uopt.retcode, objective = uopt.objective)

_, reconstruct = pack(default_vectorizer(), dynprob.ps)
optimized_parameters = reconstruct(uopt.u)
optsol, _ = layer(nothing, optimized_parameters, opt.p)
trajectoryplot(optsol)

v_relaxed = optsol[:p₂]
Δt = 12.0 / length(v_relaxed)

function sum_up_rounding(v, Δt)
    binary = zeros(length(v))
    accumulator = 0.0
    max_deviation = 0.0
    for (i, q) in enumerate(v)
        if accumulator + q * Δt >= 0.5 * Δt
            binary[i] = 1.0
            accumulator += (q - 1.0) * Δt
        else
            accumulator += q * Δt
        end
        max_deviation = max(max_deviation, abs(accumulator))
    end
    return binary, max_deviation
end
v_binary, max_deviation = sum_up_rounding(v_relaxed, Δt)

binary_parameters = copy(optimized_parameters)
binary_parameters.parameter.parameters.p₂.parameter.ps .= v_binary
binsol, _ = layer(nothing, binary_parameters, opt.p)
trajectoryplot(binsol)

(;
    relaxed_objective = uopt.objective,
    binary_objective = last(binsol.u)[3],
    max_integral_deviation = max_deviation,
    bound = 0.5 * Δt,
)

shooting_points = [3.4, 6.1, 9.75]
initial_conditions = (
    Tunable(i; kind = Val(:variable), lowerbound = 0.0, upperbound = 2.0) => shooting_points
        for i in 1:3
)
mslayer = ShootingLayer(
    prob, Tsit5(), first(control) => collect(0.25:0.25:11.75), model_constants...,
    initial_conditions...; ensemblealg = EnsembleThreads(),
    controls = [2], quadratures = [3]
)

msps, msst = LuxCore.setup(MersenneTwister(42), mslayer)

msps, msst = add_parameter_segments!(mslayer, msps, msst)

mssol, _ = mslayer(nothing, msps, msst)
trajectoryplot(mssol, show_legend = true, whiskerwidth = 10)

ms_dynprob = CorleoneDynamicOptProblem(
    mslayer, :(u₃(12.0)), ps = msps, st = msst
)

ms_opt = OptimizationProblem(ms_dynprob, AutoForwardDiff(); function_kwargs = (;))
ms_uopt = solve(
    ms_opt, Ipopt.Optimizer();
    tol = 1.0e-6, hessian_approximation = "limited-memory",
    max_iter = 1000, print_level = 0
)
@assert successful_retcode(ms_uopt)
(; retcode = ms_uopt.retcode, objective = ms_uopt.objective)

_, reconstruct_ms = pack(default_vectorizer(), ms_dynprob.ps)
ms_optimized_parameters = reconstruct_ms(ms_uopt.u)
ms_optsol, _ = mslayer(nothing, ms_optimized_parameters, ms_opt.p)
trajectoryplot(ms_optsol)

maximum(abs, shooting_constraints(ms_optsol))

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
