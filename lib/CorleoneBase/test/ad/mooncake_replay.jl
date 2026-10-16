# Keep several distinct sequential replay rules live in one process. A second
# distinct replay rule used to trip Julia 1.12's LLVM codegen; the extension now
# keeps the stage solve on a registered primitive so the derived replay rule has
# no nested derived ODE rule. This regression also pins @inferred type stability
# at the extension's replay call sites.
using Test
using CorleoneBase, CommonSolve, SciMLBase, OrdinaryDiffEqTsit5
using Mooncake, SciMLSensitivity
import DifferentiationInterface as DI

module MooncakeReplayFixtures
    using CorleoneBase, CommonSolve, SciMLBase, OrdinaryDiffEqTsit5
    include(joinpath(@__DIR__, "fixtures.jl"))
end

@testset "Mooncake replay inference" begin
    ext = Base.get_extension(CorleoneBase, :CorleoneBaseMooncakeExtension)
    for case in MooncakeReplayFixtures.CASES
        ode = ODEProblem(
            MooncakeReplayFixtures.dynamics(case), [1.2], (0.0, 1.0), [0.4];
            MooncakeReplayFixtures.SOLVE_KWARGS...
        )
        sequence = SequentialProblem(
            ode;
            transition = (seq, sol, i) -> i == 1 ? seq.problem : remake(sol.prob; u0 = sol.u[end]),
            terminal = (seq, sol, i) -> i >= 3
        )
        options = (; u0 = [1.2], p = [0.4])
        @inferred CorleoneBase.split_sequence_kwargs(sequence, options)
        @inferred ext.kwargless_problem(ode)
        forward = CommonSolve.solve(sequence, Tsit5(); options...)
        replay = @inferred ext.replayed_sequential_solve(sequence, Tsit5(), options, forward)
        @test length(replay) == 3
        @inferred ext.replayed_ode_solve(ode, Tsit5(), (;), forward[1])
    end
    ode = ODEProblem((u, p, t) -> zero(u), [1.2], (0.0, 1.0); abstol = 1.0e-11)
    sequence = SequentialProblem(
        ode;
        transition = (seq, sol, i) -> seq.problem,
        terminal = (seq, sol, i) -> i >= 513
    )
    forward = CommonSolve.solve(sequence, Tsit5())
    replay = @inferred ext.replayed_sequential_solve(sequence, Tsit5(), (;), forward)
    @test length(replay) == length(forward) == 513
end

@testset "Mooncake multi-form replay" begin
    for backend in (DI.AutoMooncake(), DI.AutoMooncakeForward())
        cases = MooncakeReplayFixtures.CASES
        for case in (cases[1], cases[3], cases[1], cases[2], cases[4])
            f = x -> MooncakeReplayFixtures.sequential_loss(case, x)
            prep = DI.prepare_gradient(f, backend, first(MooncakeReplayFixtures.INPUTS))
            for x in (MooncakeReplayFixtures.INPUTS..., first(MooncakeReplayFixtures.INPUTS))
                expected_value, expected_gradient = MooncakeReplayFixtures.reference_value_and_gradient(case, x)
                value, gradient = DI.value_and_gradient(f, backend, x)
                @test value ≈ expected_value rtol = 1.0e-9 atol = 1.0e-9
                @test gradient ≈ expected_gradient rtol = 1.0e-6 atol = 1.0e-7
                value, gradient = DI.value_and_gradient(f, prep, backend, x)
                @test value ≈ expected_value rtol = 1.0e-9 atol = 1.0e-9
                @test gradient ≈ expected_gradient rtol = 1.0e-6 atol = 1.0e-7
            end
        end
    end
end
