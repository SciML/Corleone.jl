using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test
using ForwardDiff, ReverseDiff, Zygote

@testset "Duplicated parameter values retain derivatives" begin
    prob = ODEProblem((u, p, t) -> p, [0.0], (0.0, 3.0), [2.0])
    layer = ShootingLayer(prob, Tsit5(), Tunable(1; kind = Val(:variable)) => [1.0, 2.0])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    function loss(q)
        parameter = (; lookup = (;), parameter = (; ps = q))
        input = merge(ps, (; parameter = (; parameters = (; p₁ = parameter))))
        expanded, _ = add_parameter_segments!(layer, input, st)
        return sum(expanded.parameter.parameters.p₁.parameter.ps)
    end
    @test ForwardDiff.gradient(loss, [2.0]) ≈ [3.0]
    @test ReverseDiff.gradient(loss, [2.0]) ≈ [3.0]
    @test only(Zygote.gradient(loss, [2.0])) ≈ [3.0]
end

@testset "Combined state and parameter constraint derivatives" begin
    prob = ODEProblem((u, p, t) -> p, [0.0], (0.0, 3.0), [2.0])
    layer = ShootingLayer(prob, Tsit5(), Tunable(1; kind = Val(:variable)) => [1.0, 2.0])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    ps, st = add_parameter_segments!(layer, ps, st)
    function trajectory(q)
        parameters = merge(ps, (; parameter = (; parameters = (; p₁ = (; lookup = (;), parameter = (; ps = q[1:3]))))))
        solutions = [
            (; u = [[0.0], [0.0]], t = [0.0, 1.0], prob),
            (; u = [[0.0], [0.0]], t = [1.0, 2.0], prob),
            (; u = [[0.0], [0.0]], t = [2.0, 3.0], prob),
        ]
        return Trajectory(layer, parameters, st, solutions, [q[1:0], [q[4]], [q[5]]])
    end
    loss(q) = sum(abs2, shooting_constraints(trajectory(q)))
    function inplace_loss(q)
        buffer = similar(q, get_number_of_shooting_constraints(layer, nothing, st))
        shooting_constraints!(buffer, trajectory(q))
        return sum(abs2, buffer)
    end
    q = [2.0, 5.0, 9.0, 1.0, -2.0]
    expected = [-6.0, -2.0, 8.0, 2.0, -4.0]
    @test shooting_constraints(trajectory(q)) == [1.0, -2.0, 3.0, 4.0]
    @test ForwardDiff.gradient(loss, q) ≈ expected
    @test ReverseDiff.gradient(loss, q) ≈ expected
    @test only(Zygote.gradient(loss, q)) ≈ expected
    @test ForwardDiff.gradient(inplace_loss, q) ≈ expected
    @test ReverseDiff.gradient(inplace_loss, q) ≈ expected
    function full_solve_loss(rates)
        parameters = merge(ps, (; parameter = (; parameters = (; p₁ = (; lookup = (;), parameter = (; ps = rates))))))
        traj, _ = layer(nothing, parameters, st)
        return sum(abs2, shooting_constraints(traj))
    end
    @test ForwardDiff.gradient(full_solve_loss, q[1:3]) ≈ [-2.0, 8.0, 8.0]

end
