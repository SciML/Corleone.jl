using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test, ForwardDiff, Zygote

@testset "Shared shooting parameters and values held in state" begin
    prob = ODEProblem((u, p, t) -> [p[1] * u[1], p[1] + u[1]],
        [1.0, 2.0], (0.0, 3.0), [0.2]; abstol = 1e-11, reltol = 1e-11)
    serial = ShootingLayer(prob, Tsit5(),
        Tunable(1; kind = Val(:variable)) => [1.0, 2.0],
        Constant(2; kind = Val(:variable)) => [1.0, 2.0])
    ps, st = LuxCore.setup(MersenneTwister(1), serial)
    threaded = ShootingLayer(serial.name, prob, serial.alg, EnsembleThreads(),
        serial.initialcondition, serial.parameter)
    function objective(layer, q)
        params = (;
            initialcondition = (; initialconditions = (;
                u₁ = (; lookup = (;), parameter = (; ps = q[2:3])), u₂ = (;))),
            parameter = (; parameters = (; p₁ = (; lookup = (;), parameter = (; ps = q[1:1])))))
        fixed_state = merge(st.initialcondition.initialconditions.u₂, (;
            parameters = (; lookup = (;), parameter = (; ps = [q[4], q[4]]))))
        children = merge(st.initialcondition.initialconditions, (; u₂ = fixed_state))
        ic_state = merge(st.initialcondition, (; initialconditions = children))
        states = merge(st, (; initialcondition = ic_state))
        traj, _ = layer(nothing, params, states)
        return sum(u -> sum(abs2, u), traj.u) +
            0.3 * sum(r -> sum(abs2, r), traj.residuals)
    end
    for q in ([0.2, 1.5, 2.5, 3.0], [0.3, 2.0, 3.5, 4.0])
        f = q -> objective(threaded, q)
        expected = ForwardDiff.gradient(q -> objective(serial, q), q)
        finite_difference = map(eachindex(q)) do i
            delta = zeros(length(q))
            delta[i] = 1e-5
            (f(q + delta) - f(q - delta)) / 2e-5
        end
        @test f(q) ≈ objective(serial, q)
        @test ForwardDiff.gradient(f, q) ≈ expected
        @test expected ≈ finite_difference rtol = 1e-7
    end
end

@testset "Block result cotangents" begin
    function objective(q, component)
        results = [(q[1:2] .* i, q[2:3] .* i, (; value = q[3:4] .* i)) for i in 1:3]
        solutions, residuals, state = Corleone.unpack_shooting_results(results)
        losses = (sum(x -> sum(abs2, x), solutions),
            sum(x -> sum(abs2, x), residuals), sum(abs2, state.value))
        return component == 0 ? sum(losses) : losses[component]
    end
    q = [0.2, 0.4, 0.6, 0.8]
    for component in 0:3
        f = q -> objective(q, component)
        @test only(Zygote.gradient(f, q)) ≈ ForwardDiff.gradient(f, q)
    end
end

@testset "Reverse map accumulates shared closure cotangents" begin
    function objective(q)
        vals = Corleone.shooting_map(EnsembleThreads(), collect(1:5)) do i
            q[1] * i + q[2]^2
        end
        return sum(abs2, vals)
    end
    for q in ([0.3, 1.2], [0.4, 2.0])
        @test only(Zygote.gradient(objective, q)) ≈ ForwardDiff.gradient(objective, q)
    end
end
