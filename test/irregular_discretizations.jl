using Corleone
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqBDF
using Test
using Random
using LuxCore
using Symbolics
using ForwardDiff
using ComponentArrays


shooting_points = [0.0, 1.0, 2.0]

c1 = ControlParameter([0.0], name = :c1, controls = [2.0])
c2 = ControlParameter(0.0:0.1:1.9, name = :c2, controls = LinRange(3, 5, 20))
c3 = ControlParameter(0.0:0.1:0.5, name = :c3, controls = LinRange(0, 1, 6))


controls = (c1, c2, c3)

testdyn(u, p, t) = u - [1.0, 2.0, 3.0] .* p
u0 = rand(3)
prob = ODEProblem(testdyn, u0, (0.0, 3.0), zeros(3))


layer = MultipleShootingLayer(prob, Tsit5(), [1, 2, 3], controls, shooting_points)
ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)

@testset "Repeating controls to align control and shooting discretizations" begin
    for (i, c) in enumerate(controls)
        for d in layer.duplicate_controls[i]
            @test p["layer_$(d.pre.i)"].controls[d.pre.idx] == p["layer_$(d.post.i)"].controls[d.post.idx]
        end
    end


    matching = get_shooting_constraints(layer)

    sols, _ = layer(nothing, p, st)

    num_shooting_conds = 3 * (length(shooting_points) - 1)
    num_repeat_conds = sum([length(x) for x in layer.duplicate_controls])

    @test all(matching(sols, p)[(num_shooting_conds + 1):end] .== 0.0)
end
