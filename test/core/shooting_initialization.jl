using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, Test

@testset "Shooting guesses from a reference solve" begin
    prob = ODEProblem((u, p, t) -> p .* u, [1.0, 2.0], (0.0, 3.0), [0.2, -0.1]; abstol = 1.0e-12, reltol = 1.0e-12)
    times = [0.5, 1.5, 2.0]
    reference = solve(prob, Tsit5())
    definitions = map(1:2) do i
        Tunable(i; kind = Val(:variable), default = (rng, times::AbstractVector) -> [reference(t)[i] for t in times]) => times
    end
    for alg in (EnsembleSerial(), EnsembleThreads())
        layer = ShootingLayer(prob, Tsit5(), definitions...; ensemblealg = alg)
        ps, st = LuxCore.setup(MersenneTwister(42), layer)
        for (i, entry) in enumerate(values(ps.initialcondition.initialconditions))
            @test entry.parameter.ps ≈ [prob.u0[i] * exp(prob.p[i] * t) for t in times] rtol = 1.0e-9
        end
        trajectory, _ = layer(nothing, ps, st)
        @test first(trajectory.u) == prob.u0
        @test length(shooting_constraints(trajectory)) == 6
        @test maximum(abs, shooting_constraints(trajectory)) < 1.0e-9
        @test last(trajectory.u) ≈ reference(3.0) rtol = 1.0e-9
    end
end

@testset "Mixed reference and linear guesses on separate grids" begin
    prob = ODEProblem((u, p, t) -> p .* u, [1.0, 2.0], (2.0, 6.0), [0.2, -0.1])
    reference = solve(prob, Tsit5(); abstol = 1.0e-12, reltol = 1.0e-12)
    layer = ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:variable), default = (rng, times::AbstractVector) -> [reference(t)[1] for t in times]) => [3.0, 5.0],
        Tunable(2; kind = Val(:variable), default = (rng, times::AbstractVector) -> 2 .+ (4 .- 2) .* (times .- 2) ./ (6 - 2)) => [4.0, 5.0]
    )
    ps, st = LuxCore.setup(MersenneTwister(42), layer)
    @test ps.initialcondition.initialconditions.u₁.parameter.ps ≈ exp.(0.2 .* [1.0, 3.0]) rtol = 1.0e-9
    @test ps.initialcondition.initialconditions.u₂.parameter.ps == [3.0, 3.5]
    @test get_number_of_shooting_constraints(layer, ps, st) == 4
end
