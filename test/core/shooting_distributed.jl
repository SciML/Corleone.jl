# Run through GROUP=Distributed or in an environment containing the group dependencies.
using Distributed, Test, Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, ForwardDiff

# Load definitions before entering the testset so solver calls see the current world.
Base.include(Main, joinpath(@__DIR__, "..", "shared", "shooting_ad_case.jl"))

@testset "Distributed shooting" begin
    # Workers need the same test dependencies and developed package as the parent.
    project = dirname(Base.active_project())
    added = addprocs(2; exeflags = `--project=$project --startup-file=no --compiled-modules=existing`)
    try
        @everywhere added using Corleone, LuxCore, OrdinaryDiffEqTsit5, ForwardDiff
        fixture = joinpath(@__DIR__, "..", "shared", "shooting_ad_case.jl")
        @everywhere added include($fixture)
        prob = ODEProblem(Main.ShootingADCase.scalar_dynamics, [1.0], (0.0, 3.0), [0.2])
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
        @testset "Shared parameters, tunable resets and values held in state" begin
            serial = Main.ShootingADCase.backend_case(false, EnsembleSerial())
            distributed = Main.ShootingADCase.backend_case(false, EnsembleDistributed())
            for q in ([0.2, 1.0, 1.5, 2.5, 3.0], [0.3, 1.2, 2.0, 3.5, 4.0])
                original = copy(q)
                f = distributed
                @test f(q) ≈ serial(q)
                gradient = ForwardDiff.gradient(f, q)
                @test gradient ≈ ForwardDiff.gradient(serial, q) rtol = 1.0e-8
                finite = map(eachindex(q)) do i
                    delta = zeros(length(q))
                    delta[i] = 1.0e-5
                    (f(q + delta) - f(q - delta)) / 2.0e-5
                end
                @test gradient ≈ finite rtol = 1.0e-7 atol = 1.0e-8
                @test q == original
            end
        end
        @testset "Shared-grid parameter and reset containers" begin
            serial = Main.ShootingADCase.shared_grid_case(false, EnsembleSerial())
            distributed = Main.ShootingADCase.shared_grid_case(false, EnsembleDistributed())
            for q in ([0.2, 0.4, 1.5, 2.5, 3.0, 0.1], [0.3, 0.5, 1.8, 2.9, 4.0, 0.2])
                original = copy(q)
                @test distributed(q) ≈ serial(q)
                @test distributed(q) ≈ Main.ShootingADCase.analytic_shared_grid_loss(q) rtol = 1.0e-8
                @test ForwardDiff.gradient(distributed, q) ≈ ForwardDiff.gradient(Main.ShootingADCase.analytic_shared_grid_loss, q) rtol = 1.0e-7 atol = 1.0e-8
                @test q == original
            end
        end
    finally
        rmprocs(added)
    end
end
