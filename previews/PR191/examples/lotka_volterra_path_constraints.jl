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
    w, s, c₁, c₂ = p
    du[1] = u[1] - u[1] * u[2] - c₁ * w * u[1]
    du[2] = -u[2] + u[1] * u[2] - c₂ * w * u[2]
    du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
    return nothing
end

tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [0.0, 0.0, 0.4, 0.2]
prob = ODEProblem(lotka_dynamics, u0, tspan, p0; abstol = 1.0e-9, reltol = 1.0e-9)

cgrid = collect(0.1:0.1:11.9)
control = Tunable(
    1; kind = Val(:parameter), default = 0.5,
    lowerbound = 0.0, upperbound = 1.0
) => cgrid
slack = Tunable(
    2; kind = Val(:parameter), default = 0.0,
    lowerbound = 0.0, upperbound = Inf
) => nothing
catch_coefficients = Constant([3, 4]; kind = Val(:parameter)) => nothing
layer = ShootingLayer(
    prob, Tsit5(), control, slack, catch_coefficients;
    controls = [1], quadratures = [3]
)

npoints = length(cgrid) + 2
dynprob = CorleoneDynamicOptProblem(
    layer, :(u₃(12.0) + 10 * p₂(0.0)),
    :(u₂ .- 0.1),
    :(u₁(12.0) - 1.1 + p₂(0.0));
    lcons = [zeros(npoints), [0.0]], ucons = [fill(Inf, npoints), [Inf]]
)
opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
uopt = solve(
    opt, Ipopt.Optimizer();
    tol = 1.0e-8, hessian_approximation = "limited-memory",
    max_iter = 1000, print_level = 0
)
@assert successful_retcode(uopt)

_, reconstruct = pack(default_vectorizer(), dynprob.ps)
optimized_parameters = reconstruct(uopt.u)
optsol, _ = layer(nothing, optimized_parameters, opt.p)
trajectoryplot(optsol)

(;
    objective = uopt.objective,
    slack = max(first(optsol[:p₂]), 0.0),
    terminal_prey = last(optsol.u)[1],
    minimum_predator = minimum(getindex.(optsol.u, 2)),
)

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
