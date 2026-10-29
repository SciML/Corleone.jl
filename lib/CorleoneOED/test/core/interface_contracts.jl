using CorleoneOED
using LinearAlgebra
using LuxCore
using OrdinaryDiffEqTsit5
using Random
using Test

struct TraceCriterion <: CorleoneOED.AbstractCriterion end

function (criterion::TraceCriterion)(F::Symmetric)
    return tr(F)
end

function _criterion_rhs(u, p, t)
    return p[1] .* u
end

prob = ODEProblem(_criterion_rhs, [1.0], (0.0, 1.0), [1.0])
layer = OEDLayer(
    prob, Tsit5(); params = [1],
    measurements = (ContinuousMeasurement(:weight, [0.0, 0.5], (u, p, t) -> u[1]),)
)
ps, st = LuxCore.setup(Random.default_rng(), layer)

@testset "Generic AbstractCriterion contract" begin
    value, new_state = TraceCriterion()(layer, nothing, ps, st)
    @test value isa Real
    @test new_state isa NamedTuple
end
