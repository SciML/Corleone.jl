using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test, ForwardDiff, ReverseDiff, Zygote

@testset "Initial-condition values and pass-through gradients" begin
    prob = ODEProblem((u, p, t) -> ones(2), [1.0, 2.0], (0.0, 4.0), [0.0])
    layer = InitialConditions(prob, Tunable(1) => [1.0, 3.0], Constant(2) => [2.0])
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    function objective(q)
        params = (;
            initialconditions = (;
                u₁ = (; lookup = ps.initialconditions.u₁.lookup, parameter = (; ps = q[1:2])),
                u₂ = ps.initialconditions.u₂,
            ),
        )
        states = (;
            active = st.active, initialconditions = (;
                u₁ = st.initialconditions.u₁,
                u₂ = (;
                    states = st.initialconditions.u₂.states,
                    parameters = (; lookup = (;), parameter = (; ps = [q[5]])),
                ),
            ),
        )
        sum((0.0, 1.0, 2.0, 3.0)) do t
            (u, residuals), _ = layer((q[3:4], t), params, states)
            sum(abs2, u) + sum(abs2, residuals)
        end
    end
    q = [10.0, 30.0, 4.0, 5.0, 20.0]
    # Every selected value contributes to its output and matching residual;
    # unselected incoming values remain active through the pass-through branch.
    expected = [32.0, 112.0, -48.0, 30.0, 40.0]
    @test ForwardDiff.gradient(objective, q) ≈ expected
    @test ReverseDiff.gradient(objective, q) ≈ expected
    @test only(Zygote.gradient(objective, q)) ≈ expected
end

@testset "ForwardDiff with the first lifted value after the start" begin
    prob = ODEProblem((u, p, t) -> [p[1] + zero(u[1])], [1.0], (0.0, 3.0), [1.0])
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [1.0, 2.0]
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    function endpoint(q)
        parameters = merge(
            ps, (;
                initialcondition = (;
                    initialconditions = (;
                        u₁ = (; lookup = (;), parameter = (; ps = q)),
                    ),
                ),
            )
        )
        trajectory, _ = layer(nothing, parameters, st)
        return last(trajectory.u)[1]
    end
    @test ForwardDiff.gradient(endpoint, [10.0, 20.0]) ≈ [0.0, 1.0]
end
