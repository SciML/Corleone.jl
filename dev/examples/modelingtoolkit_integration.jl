using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using ComponentArrays, ForwardDiff
using Optimization
using OptimizationMOI, Ipopt
using LuxCore, Random
using CairoMakie

@variables begin
    x(..) = 0.5, [tunable = false, bounds = (0.0, 5.0)]
    y(..) = 0.7, [tunable = false, bounds = (0.0, 5.0)]
    u(..) = 0.0, [bounds = (0.0, 1.0), input = true]
end
@constants begin
    c₁ = 0.4
    c₂ = 0.2
end

eqs = [
    D(x(t)) ~ x(t) - x(t) * y(t) - c₁ * u(t) * x(t)
    D(y(t)) ~ -y(t) + x(t) * y(t) - c₂ * u(t) * y(t)
]

costs = [
    Symbolics.Integral(t in (0.0, 12.0))(
        (x(t) - 1.0)^2 + (y(t) - 1.0)^2
    ),
]
@named lotka_volterra = System(eqs, t, costs = costs)

dynopt = CorleoneDynamicOptProblem(
    lotka_volterra, [],
    u(t) => 0.0:0.1:11.9,
    algorithm = Tsit5(),
)

optprob = OptimizationProblem(dynopt, AutoForwardDiff())

uopt = solve(
    optprob, Ipopt.Optimizer(); max_iter = 1000, tol = 5.0e-6,
    hessian_approximation = "limited-memory", print_level = 0
)
@assert successful_retcode(uopt)
(; retcode = uopt.retcode, objective = uopt.objective)

_, reconstruct = pack(default_vectorizer(), dynopt.ps)
optimized_parameters = reconstruct(uopt.u)
optsol, _ = dynopt.layer(nothing, optimized_parameters, optprob.p)
trajectoryplot(optsol)

dynopt = CorleoneDynamicOptProblem(
    lotka_volterra, [],
    u(t) => 0.0:0.25:11.75,
    algorithm = Tsit5(),
    shooting = collect(0.25:0.25:11.75),
    problem_kwargs = (; abstol = 1.0e-9, reltol = 1.0e-9)
)

optprob = OptimizationProblem(dynopt, AutoForwardDiff())

ms_uopt = solve(
    optprob, Ipopt.Optimizer(); tol = 1.0e-6,
    hessian_approximation = "limited-memory", max_iter = 300, print_level = 0
)
@assert successful_retcode(ms_uopt)
(; retcode = ms_uopt.retcode, objective = ms_uopt.objective)

_, reconstruct_ms = pack(default_vectorizer(), dynopt.ps)
ms_optimized_parameters = reconstruct_ms(ms_uopt.u)
ms_optsol, _ = dynopt.layer(nothing, ms_optimized_parameters, optprob.p)
trajectoryplot(ms_optsol)

maximum(abs, shooting_constraints(ms_optsol))

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
