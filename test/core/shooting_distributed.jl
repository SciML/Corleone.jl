# Standalone check requiring permission to open localhost worker sockets:
# julia --project=. test/core/shooting_distributed.jl
using Distributed, Test, Corleone, LuxCore, OrdinaryDiffEqTsit5, Random

@testset "Distributed shooting" begin
    # Workers need the same test dependencies and developed package as the parent.
    project = dirname(Base.active_project())
    added = addprocs(2; exeflags = `--project=$project --startup-file=no --compiled-modules=existing`)
    try
        @everywhere added using Corleone, LuxCore, OrdinaryDiffEqTsit5
        prob = ODEProblem((u, p, t) -> [p[1] * u[1]], [1.0], (0.0, 3.0), [0.2])
        serial = ShootingLayer(
            prob, Tsit5(),
            Tunable(1; kind = Val(:variable)) => [1.0, 2.0]
        )
        distributed = ShootingLayer(
            serial.name, prob, serial.alg, EnsembleDistributed(),
            serial.initialcondition, serial.parameter
        )
        ps, st = LuxCore.setup(MersenneTwister(1), serial)
        expected, expected_st = serial(nothing, ps, st)
        actual, actual_st = distributed(nothing, ps, st)
        @test actual.u ≈ expected.u
        @test actual.t == expected.t
        @test actual.residuals ≈ expected.residuals
        @test actual.reset_indices == expected.reset_indices
        @test actual_st == expected_st
        @test actual_st.timestops === st.timestops
        repeated, repeated_st = distributed(nothing, ps, actual_st)
        @test repeated.u ≈ expected.u
        @test repeated.t == expected.t
        @test repeated_st.timestops === st.timestops
    finally
        rmprocs(added)
    end
end
