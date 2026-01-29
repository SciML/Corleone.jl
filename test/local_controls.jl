using Corleone
using OrdinaryDiffEqTsit5
using Test
using Random
using LuxCore

rng = Random.default_rng()

c = ControlParameter(0:0.01:1.0)
lb, ub = Corleone.get_bounds(c)

@test c.controls === Corleone.default_u
@test c.bounds === Corleone.default_bounds
@test c.t == collect(0:0.01:1.0)
@test_nowarn Corleone.check_consistency(rng, c)
@test unique(lb) == [-Inf]
@test unique(ub) == [Inf]

c1 = ControlParameter(1.0:10.0, bounds = (0.0, 1.0))
lb1, ub1 = Corleone.get_bounds(c1)
@test unique(lb1) == [0.0]
@test unique(ub1) == [1.0]
@test_nowarn Corleone.check_consistency(rng, c1)

c2 = ControlParameter(1.0:10.0, bounds = (-ones(10), ones(10)))
lb2, ub2 = Corleone.get_bounds(c2)
@test unique(lb2) == [-1.0]
@test unique(ub2) == [1.0]
@test_nowarn Corleone.check_consistency(rng, c2)

c3 = ControlParameter(1.0:10.0, bounds = (-ones(10), ones(10)), controls = collect(0.0:0.1:0.9))
@test Corleone.get_controls(rng, c3) == collect(0.0:0.1:0.9)
@test_nowarn Corleone.check_consistency(rng, c3)

c4 = ControlParameter(1.0:10.0, bounds = (-ones(11), ones(10)), controls = collect(0.0:0.1:0.9))
@test_throws "Incompatible control bound definition" Corleone.check_consistency(rng, c4)

c5 = ControlParameter(1.0:10.0, bounds = (-ones(10), ones(10)), controls = collect(0.0:0.1:1.0))
@test_throws "Sizes are inconsistent" Corleone.check_consistency(rng, c5)

c5 = ControlParameter(1.0:10.0, bounds = (ones(10), -ones(10)), controls = collect(0.0:0.1:1.0))
@test_throws "Bounds are inconsistent" Corleone.check_consistency(rng, c5)


@testset "Correct assignment of symbols" begin
    function egerstedt(du, u, p, t)
        x, y, _ = u
        u1, u2, u3 = p
        du[1] = -x * u1 + (x + y) * u2 + (x - y) * u3
        du[2] = (x + 2 * y) * u1 + (x - 2 * y) * u2 + (x + y) * u3
        return du[3] = x^2 + y^2
    end

    tspan = (0.0, 1.0)
    u0 = [0.5, 0.5, 0.0]
    p = 1 / 3 * ones(3)

    prob = ODEProblem(egerstedt, u0, tspan, p)

    N = 20
    cgrid = collect(LinRange(tspan..., N + 1))[1:(end - 1)]
    c1 = ControlParameter(
        cgrid, name = :con1, bounds = (0.0, 1.0), controls = LinRange(0.0, 0.2, N)
    )
    c2 = ControlParameter(
        cgrid, name = :con2, bounds = (0.0, 1.0), controls = LinRange(0.3, 0.5, N)
    )
    c3 = ControlParameter(
        cgrid, name = :con3, bounds = (0.0, 1.0), controls = LinRange(0.6, 0.8, N)
    )

    layer = Corleone.SingleShootingLayer(prob, Tsit5(), controls = ([2, 3, 1] .=> [c2, c3, c1]))
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sol, _ = layer(nothing, ps, st)

    @test all(0.6 .<= sol[:con3] .<= 0.8)
    @test all(0.3 .<= sol[:con2] .<= 0.5)
    @test all(0.0 .<= sol[:con1] .<= 0.2)
end