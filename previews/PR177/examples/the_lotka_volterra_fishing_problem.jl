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
    du[1] = u[1] - p[1] * prod(u[1:2]) - 0.4 * p[2] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[2] * u[2]
    return du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
end
tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [1.0, 0.0, 1.0]
prob = ODEProblem(lotka_dynamics, u0, tspan, p0; abstol = 1.0e-9, reltol = 1.0e-9)

cgrid = collect(0.1:0.1:11.9)
control = Tunable(
    2; kind = Val(:parameter), default = 1.0,
    lowerbound = 0.0, upperbound = 1.0
) => cgrid
physical_parameters = (
    Constant([3, 1]; kind = Val(:parameter)) => nothing,
)
layer = ShootingLayer(
    prob, Tsit5(), control, physical_parameters...;
    controls = [2], quadratures = [3]
)

dynprob = CorleoneDynamicOptProblem(
    layer, :(u₃(12.0))
)
opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
uopt = solve(
    opt,
    Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300,
    print_level = 3
);
@assert successful_retcode(uopt)

_, reconstruct = pack(default_vectorizer(), dynprob.ps)
optimized_parameters = reconstruct(uopt.u)
optsol, _ = layer(nothing, optimized_parameters, opt.p)
trajectoryplot(optsol)

shooting_points = [3.4, 6.1, 9.75]
initial_conditions = (
    Tunable(i; kind = Val(:variable), lowerbound = 0.0, upperbound = 2.0) => shooting_points for i in 1:3
)
mslayer = ShootingLayer(
    prob, Tsit5(), first(control) => collect(0.25:0.25:11.75), physical_parameters...,
    initial_conditions...; ensemblealg = EnsembleThreads(),
    controls = [2], quadratures = [3]
)

msps, msst = LuxCore.setup(MersenneTwister(42), mslayer)

msps, msst = add_parameter_segments!(mslayer, msps, msst)

mssol, _ = mslayer(nothing, msps, msst);
trajectoryplot(mssol, show_legend = true, whiskerwidth = 10)

ms_dynprob = CorleoneDynamicOptProblem(
    mslayer, :(u₃(12.0)), ps = msps, st = msst
)

ms_opt = OptimizationProblem(ms_dynprob, AutoForwardDiff(); function_kwargs = (;))
ms_uopt = solve(
    ms_opt,
    Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 1000,
    print_level = 5
);
@assert successful_retcode(ms_uopt)

_, reconstruct_ms = pack(default_vectorizer(), ms_dynprob.ps)
optimized_parameters = reconstruct_ms(ms_uopt.u)
ms_optsol, _ = mslayer(nothing, optimized_parameters, ms_opt.p)
trajectoryplot(ms_optsol)

maximum(abs, shooting_constraints(ms_optsol))

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
