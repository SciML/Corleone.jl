using Corleone

using LuxCore
using Random
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationLBFGSB
using ForwardDiff
using ComponentArrays
using SymbolicIndexingInterface
using CairoMakie
Makie.inline!()

function lqr_dynamics(x, p, t)
    a, b, u = p
    du = a .* x[1:(end - 1)] .+ b .* u
    costs = 10.0 .* (x[1] .- 3.0) .^ 2 .+ 0.1 .* u .^ 2
    return vcat(du, costs)
end

tspan = (0.0, 12.0)
u0 = [1.0, 2.0, 0.0]
p0 = [-1.0, 1.0, 0.0]
lqr_problem = ODEProblem(lqr_dynamics, u0, tspan, p0)

control_grid = collect(0.1:0.1:11.9)
predictor = ShootingLayer(
    lqr_problem, Tsit5(),
    Tunable(3; kind = Val(:parameter), default = (rng, n) -> rand(rng, n)) => control_grid,
    Constant(1; kind = Val(:parameter)) => nothing,
    Constant(2; kind = Val(:parameter)) => nothing;
    controls = [3], quadratures = [3],
)

ps, st = LuxCore.setup(MersenneTwister(42), predictor)
solution, _ = predictor(nothing, ps, st)
trajectoryplot(solution)

dynprob = CorleoneDynamicOptProblem(
    predictor, :(u₃(12.0)); ps, st
)

opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
usol = solve(opt, OptimizationLBFGSB.LBFGSB())
@assert successful_retcode(usol)
(; retcode = usol.retcode, objective = usol.objective)

_, reconstruct = pack(default_vectorizer(), dynprob.ps)
optimized_parameters = reconstruct(usol.u)
optimized_solution, _ = predictor(nothing, optimized_parameters, opt.p)
trajectoryplot(optimized_solution)

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
