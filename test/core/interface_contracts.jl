using Corleone, LuxCore, OrdinaryDiffEqTsit5, StableRNGs, ComponentArrays, Test

@testset "Native Lux layer contract" begin
    prob = ODEProblem((u, p, t) -> p[1] .* u, [1.0], (0.0, 1.0), [0.2])
    for alg in (EnsembleSerial(), EnsembleThreads()), lifted in (false, true)
        definitions = lifted ? (Tunable(1; kind = Val(:variable)) => [0.5],) : ()
        layer = ShootingLayer(prob, Tsit5(), definitions...; ensemblealg = alg)
        ps, st = LuxCore.setup(StableRNG(901), layer)
        @test LuxCore.initialparameters(StableRNG(901), layer) == ps
        @test LuxCore.initialstates(StableRNG(901), layer) == st
        @test LuxCore.parameterlength(layer) == length(ComponentArray(ps)) == (lifted ? 2 : 1)
        @test LuxCore.statelength(layer) >= 0
        for parameters in (ps, ComponentArray(ps))
            trajectory, next_state = LuxCore.apply(layer, nothing, parameters, st)
            @test trajectory isa Trajectory
            @test next_state isa NamedTuple
            @test trajectory.t == (lifted ? [0.0, 0.5, 0.5, 1.0] : [0.0, 1.0])
            @test trajectory.ps === parameters
            @test trajectory.st === next_state
            @test length(shooting_constraints(trajectory)) == get_number_of_shooting_constraints(layer, parameters, st)
            again, _ = layer(nothing, parameters, next_state)
            @test again.u ≈ trajectory.u
            @test again.residuals ≈ trajectory.residuals
        end
    end
end
