using Test, Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, ForwardDiff, ReverseDiff, Zygote
using RuntimeGeneratedFunctions

@testset "Symbolic facade getters preserve state and parameter gradients" begin
    prob = ODEProblem((u, p, t) -> [p[1]], [1.0], (0.0, 1.0), [0.5])
    layer = ShootingLayer(prob, Tsit5())
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    for evaluator in (Val(:Eval), Val(:RuntimeGeneratedFunctions))
        d = CorleoneDynamicOptProblem(
            layer, :(u₁(1.0) + p₁(1.0)),
            :([u₁(0.0) + p₁(0.0), u₁(1.0) + p₁(1.0)]); ps, st, evaluator, lcons = fill(-Inf, 2), ucons = fill(Inf, 2)
        )
        function trajectory(q)
            parameters = (;
                initialcondition = ps.initialcondition,
                parameter = (; parameters = (; p₁ = (; lookup = (;), parameter = (; ps = q[3:3])))),
            )
            sols = [(; u = [[q[1]], [q[2]]], t = [0.0, 1.0], prob)]
            Trajectory(layer, parameters, st, sols, [similar(q, 0)])
        end
        loss(q) = d.objective(trajectory(q))
        constraints(q) = sum(abs2, only(d.constraints)(trajectory(q)))
        for q in ([1.0, 2.0, 0.5], [2.0, 3.0, 1.0])
            expected = [2 * (q[1] + q[3]), 2 * (q[2] + q[3]), 2 * (q[1] + q[2] + 2q[3])]
            for gradient in (ForwardDiff.gradient, ReverseDiff.gradient, (f, x) -> only(Zygote.gradient(f, x)))
                @test gradient(loss, q) ≈ [0.0, 1.0, 1.0]
                @test gradient(constraints, q) ≈ expected
            end
        end
    end
end
