using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, ReverseDiff, Test

@testset "Differentiation through trajectory assembly and parameter histories" begin
    prob = ODEProblem((u, p, t) -> [p[1]], [1.0], (0.0, 2.0), [2.0])
    layer = ShootingLayer(prob, Tsit5(), 1 => [1.0])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    with_rates(q) = (;
        initialcondition = ps.initialcondition,
        parameter = (; parameters = (; p₁ = (; lookup = (;), parameter = (; ps = q)))),
    )
    function trajectory(q)
        # Isolate assembly from the ODE solver's own AD support. The two
        # segments share a continuous boundary that assembly must reduce.
        solutions = [
            (; u = [[q[1]], [q[1] + q[2]]], t = [0.0, 1.0], prob),
            (; u = [[q[1] + q[2]], [q[1] + 2q[2]]], t = [1.0, 2.0], prob),
        ]
        return Trajectory(layer, with_rates(q), st, solutions, [Float64[], Float64[]])
    end
    @test ReverseDiff.gradient(q -> last(trajectory(q).u)[1], [2.0, 4.0]) ≈ [1.0, 2.0]
    @test ReverseDiff.gradient(q -> sum(trajectory(q)[:p₁]), [2.0, 4.0]) ≈ [1.0, 1.0]
end

# Exercise the public constructor and the layer's named builder separately: a
# broad downstream constructor rule previously intercepted only the former.
using Zygote

@testset "Assembly gradients across backends and reset boundaries" begin
    for reset in (false, true)
        prob = ODEProblem((u, p, t) -> [p[1]], [1.0], (0.0, 2.0), [2.0])
        defs = reset ? (1.0 => (; variables = (1 => 1.0,)),) : ()
        layer = ShootingLayer(prob, Tsit5(), defs..., 1 => [1.0])
        ps, st = LuxCore.setup(MersenneTwister(1), layer)
        with_rates(q) = merge(ps, (; parameter = (; parameters = (; p₁ = (; lookup = (;), parameter = (; ps = q))))))
        function objective(q, assembly, mode)
            solutions = [
                (; u = [[q[1]], [q[1] + q[2]]], t = [0.0, 1.0], prob),
                (; u = [[q[1] + q[2]], [q[1] + 2q[2]]], t = [1.0, 2.0], prob),
            ]
            traj = assembly(layer, with_rates(q), st, solutions, [[q[1] - q[2]], Float64[]])
            mode == :state && return sum(sum, traj.u)
            mode == :parameters && return sum(traj[:p₁])
            mode == :residuals && return sum(r -> sum(abs2, r), traj.residuals)
            return sum(sum, traj.u) + sum(traj[:p₁]) + sum(r -> sum(abs2, r), traj.residuals)
        end
        for assembly in (Trajectory, Corleone.build_trajectory),
                mode in (:state, :parameters, :residuals, :combined), q in ([2.0, 4.0], [3.0, 1.0])
            state_gradient = fill(reset ? 4.0 : 3.0, 2)
            residual_gradient = 2(q[1] - q[2]) .* [1.0, -1.0]
            expected = mode == :state ? state_gradient : mode == :parameters ? ones(2) :
                mode == :residuals ? residual_gradient : state_gradient + ones(2) + residual_gradient
            f = q -> objective(q, assembly, mode)
            @test ReverseDiff.gradient(f, q) ≈ expected
            @test only(Zygote.gradient(f, q)) ≈ expected
        end
    end
    # Repeated source indices must accumulate, rather than overwrite, adjoints.
    f(q) = sum(sum, Corleone.gather_trajectory_states([[q]], [1, 1], [1, 1]))
    @test only(Zygote.gradient(f, [2.0, 4.0])) == [2.0, 2.0]
end
