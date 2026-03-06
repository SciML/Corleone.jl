using Corleone
using OrdinaryDiffEqTsit5
using Test
using Random
using LuxCore

rng = Random.default_rng()

c = ControlParameter(0:0.01:1.0)
lb, ub = Corleone.get_bounds(c)
ps, st = LuxCore.setup(rng, c)

@test ps == zero(c.t)
@test lb == collect(c.t) .- Inf 
@test ub == collect(c.t) .+ Inf
@test c.t == collect(0:0.01:1.0)
@test length(ps) == length(c.t)

c = ControlParameter(0:0.01:1.0, name = :test, controls = (rng, t) -> rand(rng, length(t)), bounds = (t) -> (-ones(length(t)), ones(length(t))))
lb, ub = Corleone.get_bounds(c)
ps, st = LuxCore.setup(rng, c)

@test lb <= ps <= ub
@test lb == zero(ps) .- 1.0
@test ub == zero(ps) .+ 1.0
@test c.t == collect(0:0.01:1.0)
@test length(ps) == length(c.t)

c = ControlParameter(0:0.1:1.0, name = :test2, controls = (rng, t) -> [randn(rng, 3) for i in eachindex(t)])
lb, ub = Corleone.get_bounds(c)
ps, st = LuxCore.setup(rng, c)

@test lb <= ps <= ub
@test eltype(lb) == eltype(ub) == eltype(ps)
@test c.t == collect(0:0.1:1.0)
@test length(ps) == length(c.t)

c = ControlParameter([], name = :test3, controls = (rng, t) -> [randn(rng, 3)])
lb, ub = Corleone.get_bounds(c)
ps, st = LuxCore.setup(rng, c)
@test c(0., ps, st) == c(10., ps, st)
@test lb <= ps <= ub
@test isempty(c.t)
@test length(ps) == 1



controls = ControlParameters(
    :u => 0.0:0.1:10.0,
    ControlParameter(0.0:0.2:10.0, name = :v, controls = (rng, t) -> rand(rng, length(t)), bounds = t -> (zeros(length(t)), ones(length(t))));
    transform = (cs) -> (u = (1, cs.u), v = cs.v)
)
ps, st = LuxCore.setup(rng, controls)
controls(0.0, ps, st)

@code_warntype controls(0.0, ps, st)


#= @testset "Correct assignment of symbols" begin
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
 =#