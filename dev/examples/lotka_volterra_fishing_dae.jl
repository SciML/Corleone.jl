using Corleone

using LuxCore
using Random
using OrdinaryDiffEqRosenbrock
using SymbolicIndexingInterface
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using ForwardDiff
using LinearAlgebra
using CairoMakie
using Corleone: SciMLBase
Makie.inline!()

function lotka_dae(du, u, p, t)
    x, y, z, w, cost = u
    v, c₁, c₂ = p
    du[1] = x * z
    du[2] = y * w
    du[3] = z - (1.0 - y - c₁ * v)
    du[4] = w - (-1.0 + x - c₂ * v)
    du[5] = (x - 1.0)^2 + (y - 1.0)^2
    return nothing
end

tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.3, -0.5, 0.0]
p0 = [0.0, 0.4, 0.2]

prob = ODEProblem(
    ODEFunction(lotka_dae; mass_matrix = diagm([1.0, 1.0, 0.0, 0.0, 1.0])),
    u0, tspan, p0; abstol = 1.0e-9, reltol = 1.0e-9,
    initializealg = SciMLBase.NoInit()
)

cgrid = collect(0.1:0.1:11.9)
control = Tunable(
    1; kind = Val(:parameter), default = 0.0,
    lowerbound = 0.0, upperbound = 1.0
) => cgrid
catch_coefficients = Constant([2, 3]; kind = Val(:parameter)) => nothing
algebraic_initial_conditions = (
    Tunable(i; kind = Val(:variable)) => nothing for i in 3:4
)
layer = ShootingLayer(
    prob, Rodas5P(), control, catch_coefficients, algebraic_initial_conditions...;
    controls = [1], quadratures = [5]
)

dae_initialization = (
    :(u₃(0.0) - (1.0 - u₂(0.0) - p₂(0.0) * p₁(0.0))),
    :(u₄(0.0) - (-1.0 + u₁(0.0) - p₃(0.0) * p₁(0.0))),
)

dynprob = CorleoneDynamicOptProblem(
    layer, :(u₅(12.0)), dae_initialization...;
    lcons = [0.0, 0.0], ucons = [0.0, 0.0]
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
    retcode = uopt.retcode,
    objective = uopt.objective,
    initialization_residual = maximum(
        abs, optimization_constraints(
            dynprob, optimized_parameters, opt.p
        )
    ),
)

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
