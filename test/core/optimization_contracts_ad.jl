using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test
using ForwardDiff, ReverseDiff, Zygote

@testset "Shooting constraint value derivatives" begin
    prob = ODEProblem((u, p, t) -> [p[1] + zero(u[1])], [1.0], (0.0, 2.0), [1.0])
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [0.0, 1.0]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    function trajectory(q)
        solutions = [
            (; u = [[q[1]], [q[1] + q[3]]], t = [0.0, 1.0], prob),
            (; u = [[q[2]], [q[2] + q[3]]], t = [1.0, 2.0], prob),
        ]
        return Trajectory(
            layer, ps, st, solutions,
            [eltype(q)[], [q[2] - q[1] - q[3]]]
        )
    end
    loss(q) = sum(abs2, shooting_constraints(trajectory(q)))
    inplace_loss(q) = begin
        res = similar(q, get_number_of_shooting_constraints(layer, nothing, st))
        shooting_constraints!(res, trajectory(q))
        sum(abs2, res)
    end
    for q in ([2.0, 5.0, 0.5], [3.0, 2.0, 1.5])
        r = q[2] - q[1] - q[3]
        expected = [-2r, 2r, -2r]
        @test ForwardDiff.gradient(loss, q) ≈ expected
        @test ReverseDiff.gradient(loss, q) ≈ expected
        @test only(Zygote.gradient(loss, q)) ≈ expected
        @test ForwardDiff.gradient(inplace_loss, q) ≈ expected
    end

    function full_solve_loss(q)
        variable = merge(
            ps.initialcondition.initialconditions.u₁,
            (; parameter = (; ps = q))
        )
        parameters = merge(
            ps, (;
                initialcondition = (;
                    initialconditions = (; u₁ = variable),
                ),
            )
        )
        traj, _ = layer(nothing, parameters, st)
        return sum(abs2, shooting_constraints(traj))
    end
    @test ForwardDiff.gradient(full_solve_loss, [2.0, 5.0]) ≈ [-4.0, 4.0]
end
