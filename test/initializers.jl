using Corleone
using LuxCore
using Random
using SciMLBase
using Test

rng = MersenneTwister(123)

f!(du, u, p, t) = begin
    du[1] = -p[1] * u[1]
    du[2] = u[1] - u[2]
    du[3] = u[1] + u[2]
    nothing
end

prob = ODEProblem(f!, [1.0, 2.0, 3.0], (0.0, 5.0), [0.25])

@testset "InitialCondition constructor and bounds" begin
    layer = InitialCondition(prob; name = :ic, tunable_ic = [1, 3], quadrature_indices = [2])
    @test layer.name == :ic
    @test layer.tunable_ic == [1, 3]
    @test layer.quadrature_indices == [2]
    lb = Corleone.get_lower_bound(layer)
    ub = Corleone.get_upper_bound(layer)
    @test lb == fill(-Inf, 2)
    @test ub == fill(Inf, 2)
    bounds_layer = InitialCondition(
        prob;
        tunable_ic = [1, 3],
        bounds_ic = t0 -> ([ t0 - 10.0,1.0, -3.0], [ t0 + 10.0,1.0, 3.0]),
    )
    @test Corleone.get_lower_bound(bounds_layer) == [-10.0, -3.0]
    @test Corleone.get_upper_bound(bounds_layer) == [10.0, 3.0]
    @test_throws AssertionError InitialCondition(prob; tunable_ic = [4])
    @test_throws AssertionError InitialCondition(prob; quadrature_indices = [4])
    @test_throws AssertionError InitialCondition(prob; tunable_ic = [1, 2], quadrature_indices = [2])
end

@testset "InitialCondition setup and call" begin
    layer = InitialCondition(prob; tunable_ic = [1, 3], quadrature_indices = [2])
    ps = @inferred LuxCore.initialparameters(rng, layer)
    @test ps == [1.0, 3.0]
    ps[1] = 99.0
    @test layer.problem.u0[1] == 1.0
    @test @inferred(LuxCore.parameterlength(layer)) == 2
    st = @inferred LuxCore.initialstates(rng, layer)
    @test st.u0 == [1.0, 2.0, 3.0]
    @test st.keeps == [false, true, false]
    @test st.replaces == Bool[
        1 0
        0 0
        0 1
    ]
    @test st.quadrature_indices == [2]
    pnew = [10.0, 30.0]
    remade_prob, st2 = @inferred layer(nothing, pnew, st)
    @test remade_prob.u0 == [10.0, 2.0, 30.0]
    @test st2 == st
end

@testset "InitialCondition remake" begin
    layer = InitialCondition(
        prob;
        name = :base,
        tunable_ic = [1, 3],
        quadrature_indices = [2],
        bounds_ic = t0 -> (fill(t0 - 1.0, 2), fill(t0 + 1.0, 2)),
    )
    remade = @inferred InitialCondition SciMLBase.remake(
        layer;
        name = :remade,
        tunable_ic = [3],
        quadrature_indices = [1, 2],
        u0 = [7.0, 8.0, 9.0],
        tspan = (1.0, 2.0),
    )
    @test remade isa InitialCondition
    @test remade.name == :remade
    @test remade.tunable_ic == [3]
    @test remade.quadrature_indices == [1, 2]
    @test remade.problem.u0 == [7.0, 8.0, 9.0]
    @test remade.problem.tspan == (1.0, 2.0)
    @test remade.bounds_ic === layer.bounds_ic
    # Unsupported kwargs should be ignored by the DEProblem remake forwarding.
    remade_ignored = SciMLBase.remake(layer; unsupported_kwarg = :ignore_me)
    @test remade_ignored.problem.u0 == layer.problem.u0
    @test remade_ignored.problem.tspan == layer.problem.tspan
end
