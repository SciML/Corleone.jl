using Corleone: SciMLBase
using Corleone
using CorleoneOED
using OrdinaryDiffEqBDF
using OrdinaryDiffEqNonlinearSolve
using ComponentArrays
using LuxCore
using StableRNGs

using Test
using Optimization
using OptimizationMOI
using Ipopt
using LinearAlgebra
using SymbolicIndexingInterface

rng = StableRNG(1111)
function robertson!(out, du, u, p, t)
    k₁, k₂, k₃ = p
    x₁, x₂, x₃ = u
    out[1] = -k₁ * x₁ + k₃ * x₂ * x₃ - du[1]
    out[2] = k₁ * x₁ - k₃ * x₂ * x₃ - k₂ * x₂^2 - du[2]
    out[3] = x₁ + x₂ + x₃ - 1  # algebraic: no du term
    return
end

du0 = [-0.04, 0.04, 0.0]
u0 = [1.0, 0.0, 0.0]
differential_vars = [true, true, false]
sys = SymbolCache([:x1, :x2, :x3], [:k1, :k2, :k3], :t)
f = DAEFunction(robertson!; sys = sys)
prob = DAEProblem(f, du0, u0, (0.0, 10), [0.04, 3.0e7, 1.0e4]; abstol = 1.0e-8, reltol = 1.0e-8, differential_vars, initializealg = BrownFullBasicInit())

# Specify measurement
lb, ub = 1.0e-3, 0.95 * last(prob.tspan)
tpoints = exp.(collect(0.0:0.1:1.0) .* (log.(ub) .- log.(lb)) .+ log.(lb))
measurement = DiscreteMeasurement(:w1, tpoints, (u, p, t) -> u[1:1])

# Dummy control for now as OEDLayer without controls crashes. TODO: Fix this
pc1 = PiecewiseParameter(:k2, [0.0], 3.0e7, (3.0e7, 3.0e7))

oed = @test_nowarn  OEDLayer(
    prob, [], [1, 2, 3], pc1;
    algorithm = DFBDF(),
    measurements = [
        measurement,
    ],
)

ps, st = LuxCore.setup(rng, oed)
@test_nowarn @inferred first(oed(nothing, ps, st))

sol_layer = first(oed(nothing, ps, st))
sol_prob = solve(oed.augmented_prob, DFBDF())

@test isapprox(last(sol_prob.u), last(sol_layer.u)[1:length(oed.augmented_prob.u0)], atol = 1.0e-6)
