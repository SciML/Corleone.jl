using Test, Corleone, LuxCore, Random, ComponentArrays, OrdinaryDiffEqTsit5
using TransformVariables: asℝ₊
using ForwardDiff, ReverseDiff, Zygote

@testset "AD through generated mixed experiment preparation" begin
    problem = ODEProblem((u, p, t) -> [-p[1] * u[1] + p[2]], [1.0], (0.0, 1.0), [0.7, 0.2])
    a = (annotations = (Tunable(1; kind = Val(:variable)) => [0.0], Tunable(2) => nothing),)
    cases = (
        (
            (
                annotations = (
                    Tunable(1; kind = Val(:variable), default = [2.0, 1.0]) => [0.0, 0.5],
                    Constant(2; default = 0.3) => nothing,
                ),
            ), [2.8, 2.0, 0.4, 4.0, 0.0], 6.11,
        ),
        (
            (
                annotations = (
                    Tunable(1; kind = Val(:variable), default = 2.0, transform = asℝ₊) => [0.0],
                    Tunable(2; default = 1.1, transform = asℝ₊) => nothing,
                ),
            ), [2.8, 2.0, 0.4, 8.0, 2.42], 7.23,
        ),
    )
    for (b, expected, value) in cases
        experiments = Experiments(problem; controls = [2], experiments = (; a, b))
        ps, st = LuxCore.setup(Xoshiro(1), experiments)
        q, reconstruct = pack(default_vectorizer(), ps)
        function loss(q)
            calls, _ = experiments(nothing, reconstruct(q), st)
            return sum(
                map(calls, values(experiments.configurations)) do (parameters, state), config
                    p, _ = config.parameter(0.0, parameters.parameter, state.parameter)
                    (u, _), _ = config.initialcondition(([0.0], 0.0), parameters.initialcondition, state.initialcondition)
                    sum(abs2, p) + sum(abs2, u)
                end
            )
        end
        @test loss(q) ≈ value
        @test ForwardDiff.gradient(loss, q) ≈ expected
        @test ReverseDiff.gradient(loss, q) ≈ expected
        @test only(Zygote.gradient(loss, q)) ≈ expected
    end
end
