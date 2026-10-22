# More than 100 control intervals split the solve into several sequential segments, so the
# state is handed from one segment's solution to the next segment's initial value.
using Corleone, ComponentArrays, LuxCore, OrdinaryDiffEqTsit5, ReverseDiff, SciMLBase
using StaticArrays, Random, Test

const DT = 0.01
const N_INTERVALS = 101
control_grid() = ControlParameter(
    collect(0:(N_INTERVALS - 1)) .* DT; controls = fill(0.2, N_INTERVALS), name = :rate
)

@testset "Carried state keeps the container type of u0" begin
    x = [1.0, 2.0, 0.2]
    for u0 in (SizedVector{2}([0.0, 0.0]), MVector(0.0, 0.0), [0.0, 0.0])
        y = Corleone._carry_state(u0, x)
        @test y isa typeof(u0)
        @test y == [1.0, 2.0]
    end
    tracked = Corleone._carry_state([0.0, 0.0], ReverseDiff.track(x))
    @test tracked isa Vector
    @test length(tracked) == 2
end

@testset "Out-of-place SizedVector state keeps its container across segments" begin
    rhs(u, p, t) = SizedVector{2}([p[1] * u[1], -p[1] * u[2]])
    u0 = SizedVector{2}([1.0, 2.0])
    prob = ODEProblem(rhs, u0, (0.0, N_INTERVALS * DT), [0.2]; abstol = 1.0e-11, reltol = 1.0e-11)
    layer = SingleShootingLayer(prob, Tsit5(); controls = (1 => control_grid(),))
    ps, st = LuxCore.setup(MersenneTwister(10), layer)
    sol, _ = layer(nothing, ps, st)
    rate_integral = 0.2 * N_INTERVALS * DT
    @test last(sol.u)[1:2] ≈ [exp(rate_integral), 2exp(-rate_integral)]
end

@testset "ReverseDiff gradient through an in-place multi-segment solve" begin
    function rhs!(du, u, p, t)
        du[1] = p[1] * u[1]
        return nothing
    end
    prob = ODEProblem(
        rhs!, [1.0], (0.0, N_INTERVALS * DT), [0.2];
        abstol = 1.0e-11, reltol = 1.0e-11, sensealg = SciMLBase.NoAD()
    )
    layer = SingleShootingLayer(prob, Tsit5(); controls = (1 => control_grid(),))
    ps, st = LuxCore.setup(MersenneTwister(10), layer)
    p = ComponentArray(ps)
    objective = p -> last(first(layer(nothing, p, st)).u)[1]
    # u' = c(t) u with piecewise-constant c: u(T) = exp(DT * sum(c)), ∂u(T)/∂cᵢ = DT * u(T).
    expected = exp(DT * sum(p.controls))
    @test objective(p) ≈ expected rtol = 1.0e-9
    g = ReverseDiff.gradient(objective, p)
    @test all(isapprox.(g.controls, DT * expected; rtol = 1.0e-8))
end
