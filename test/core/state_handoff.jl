using Corleone, ComponentArrays, LuxCore, OrdinaryDiffEqTsit5, ForwardDiff, ReverseDiff, SciMLBase
using StaticArrays, Random, Test
using DiffEqBase: SensitivityADPassThrough

const DT = 0.01
const N_INTERVALS = 101

@testset "State handoff across many control intervals" begin
    for u0 in ([1.0, 2.0], SizedVector{2}([1.0, 2.0]), MVector(1.0, 2.0))
        rhs(u, p, t) = typeof(u)([p[1] * u[1], -p[1] * u[2]])
        prob = ODEProblem(rhs, u0, (0.0, N_INTERVALS * DT), [0.2]; abstol = 1.0e-11, reltol = 1.0e-11)
        layer = ShootingLayer(prob, Tsit5(), Tunable(1) => collect(1:(N_INTERVALS - 1)) .* DT)
        ps, st = LuxCore.setup(MersenneTwister(10), layer)
        trajectory, _ = layer(nothing, ps, st)
        @test length(trajectory.t) == N_INTERVALS + 1
        @test isempty(trajectory.reset_indices)
        @test all(isempty, trajectory.residuals)
        @test all(u -> u isa typeof(u0), trajectory.u)
        @test last(trajectory.u) ≈ [exp(0.2 * N_INTERVALS * DT), 2exp(-0.2 * N_INTERVALS * DT)] rtol = 1.0e-9
    end
end

@testset "Tracked state handoff through an in-place solve" begin
    function rhs!(du, u, p, t)
        du[1] = p[1] * u[1]
        return nothing
    end
    prob = ODEProblem(
        ODEFunction{true, SciMLBase.FullSpecialize}(rhs!), [1.0], (0.0, N_INTERVALS * DT), [0.2];
        sensealg = SensitivityADPassThrough(), abstol = 1.0e-11, reltol = 1.0e-11
    )
    layer = ShootingLayer(prob, Tsit5(), Tunable(1) => collect(1:(N_INTERVALS - 1)) .* DT)
    ps, st = LuxCore.setup(MersenneTwister(10), layer)
    q, reconstruct = pack(Val(:ComponentArrays), ps)
    objective(q) = last(first(layer(nothing, reconstruct(q), st)).u)[1]
    for values in (q, q .+ 0.01)
        expected = exp(DT * sum(values))
        @test objective(values) ≈ expected rtol = 1.0e-9
        for gradient in (ForwardDiff.gradient, ReverseDiff.gradient)
            @test gradient(objective, values) ≈ fill(DT * expected, N_INTERVALS) rtol = 1.0e-8
        end
    end
end
