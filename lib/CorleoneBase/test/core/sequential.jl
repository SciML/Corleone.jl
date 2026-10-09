using CorleoneBase
using CorleoneBase: retcode
using CommonSolve: CommonSolve, init, solve, solve!, step!
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: SciMLBase, ODEProblem, isinplace, remake, successful_retcode
using Test

@test realpath(pkgdir(CorleoneBase)) == realpath(joinpath(@__DIR__, "../.."))
@test Base.get_extension(CorleoneBase, :CorleoneBaseZygoteExtension) === nothing

# The analytic fixtures use only SciMLBase and a solver, not DI or AD packages.
include("../ad/fixtures.jl")

@testset "Analytic multi-stage solutions" begin
    for case in CASES, x in INPUTS
        solutions = sequential_solutions(case, x)
        expected, _ = reference_endpoints(case, x)
        @test length(solutions) == 3
        for (i, sol) in enumerate(solutions)
            @test successful_retcode(sol)
            @test isinplace(sol.prob) == is_inplace(case)
            @test sol.prob.tspan == (Float64(i - 1), Float64(i))
            @test sol.t == [Float64(i - 1), Float64(i)]
            @test sol.prob.p == [x[i == 1 ? 2 : 3]]
            @test sol.prob.u0 == (i == 1 ? [x[1]] : i == 2 ? solutions[1].u[end] : [x[4]])
            @test isapprox(sol.u[end][1], expected[i]; atol = 1.0e-9, rtol = 1.0e-9)
        end
    end
end

@testset "Initialization, stepping, and identity" begin
    template = ODEProblem((u, p, t) -> -p .* u, [9.0], (0.0, 1.0), 4.0)
    calls = Int[]
    problem = SequentialProblem(
        template;
        transition = (sol, i) -> begin
            push!(calls, i)
            remake(sol.prob; u0 = sol.u[end], tspan = (sol.t[end], sol.t[end] + 1))
        end,
        terminal = (sol, i) -> i >= 3,
    )
    it = init(problem, Tsit5(); SOLVE_KWARGS..., u0 = [1.2], p = 0.4, tspan = (2.0, 3.0))
    @test isempty(calls)
    @test it.state == 1
    @test length(it.buffer) == 1
    @test !Base.isdone(it)
    @test it.problem.u0 == [1.2]
    @test it.problem.p == 0.4
    @test it.problem.tspan == (2.0, 3.0)
    @test template.u0 == [9.0]
    @test template.p == 4.0
    @test template.tspan == (0.0, 1.0)
    buffer = it.buffer
    @test step!(it)
    @test it.state == 2
    @test calls == [2]
    w = solve!(it)
    @test w isa SolutionWrapper
    @test w.buffer === buffer
    @test it.buffer === buffer
    @test calls == [2, 3]
    @test Base.isdone(it)
    @test it.state == 3
    @test length(w) == 3
    @test w[3].u[end][1] ≈ 1.2 * exp(-1.2)
    @test retcode(w) == SciMLBase.ReturnCode.Success
    @test successful_retcode(w)
    w2 = solve!(it)
    @test w2 isa SolutionWrapper
    @test length(w2) == 3
    @test calls == [2, 3]

    single = solve(SequentialProblem(template), Tsit5(); SOLVE_KWARGS...)
    @test single isa SolutionWrapper
    @test length(single) == 1
    @test retcode(single) == SciMLBase.ReturnCode.Success
    @test successful_retcode(single)
end

# Controlled return codes exercise failures without relying on a solver's
# adaptive-step heuristics. These types and methods belong to this test module.
struct StageProblem <: SciMLBase.AbstractSciMLProblem
    succeeds::Bool
end
struct StageSolution
    succeeds::Bool
    retcode::SciMLBase.ReturnCode.T
end
StageSolution(succeeds::Bool) = StageSolution(
    succeeds, succeeds ? SciMLBase.ReturnCode.Success : SciMLBase.ReturnCode.MaxIters
)
struct NoSolve <: SciMLBase.AbstractSciMLAlgorithm end

SciMLBase.remake(p::StageProblem; kwargs...) = p
CommonSolve.solve(p::StageProblem, ::NoSolve; kwargs...) = StageSolution(p.succeeds)
SciMLBase.successful_retcode(sol::StageSolution) = sol.succeeds



@testset "Return codes, failures, and preallocated buffers" begin
    # An unsuccessful first stage produces a one-element failure result instead
    # of the earlier initialization exception, and never advances.
    w = solve(SequentialProblem(StageProblem(false)), NoSolve())
    @test w isa SolutionWrapper
    @test length(w) == 1
    @test retcode(w) == SciMLBase.ReturnCode.MaxIters
    @test !successful_retcode(w)
    @test w[1].retcode == SciMLBase.ReturnCode.MaxIters

    for preallocate in (false, true), fail_at in (0, 2, 3)
        problem = SequentialProblem(
            StageProblem(true);
            transition = (sol, i) -> StageProblem(i != fail_at),
            terminal = (sol, i) -> i >= 3,
        )
        it = init(problem, NoSolve())
        if preallocate
            append!(it.buffer, [StageSolution(false), StageSolution(false)])
        end
        buffer = it.buffer
        w = solve!(it)
        @test w isa SolutionWrapper
        @test w.buffer === buffer
        expected_state = fail_at == 0 ? 3 : fail_at
        @test it.state == expected_state
        @test length(w) == expected_state
        # Preallocated, unused slots are hidden from the wrapper.
        @test length(w.buffer) == (preallocate ? 3 : expected_state)
        for i in 1:(expected_state - (fail_at == 0 ? 0 : 1))
            @test w[i].succeeds
        end
        if fail_at != 0
            @test !w[expected_state].succeeds
            @test retcode(w) == SciMLBase.ReturnCode.MaxIters
            @test !successful_retcode(w)
            # No further stages were attempted after the failure.
        else
            @test successful_retcode(w)
            @test retcode(w) == SciMLBase.ReturnCode.Success
        end
        @test_throws BoundsError w[expected_state + 1]
    end
end

@testset "SolutionWrapper array interface" begin
    problem = SequentialProblem(
        StageProblem(true);
        transition = (sol, i) -> StageProblem(true),
        terminal = (sol, i) -> i >= 3,
    )
    it = init(problem, NoSolve())
    append!(it.buffer, [StageSolution(false), StageSolution(false)])
    w = solve!(it)
    @test w isa CorleoneBase.SolutionWrapper
    @test w isa AbstractVector
    @test length(w) == 3
    @test size(w) == (3,)
    @test firstindex(w) == 1
    @test lastindex(w) == 3
    @test eltype(w) == StageSolution
    @test Base.IndexStyle(typeof(w)) == IndexLinear()
    @test collect(w) == w.buffer[1:3]
    @test [s for s in w] == w.buffer[1:3]
    @test w[2].succeeds
    @test w[end].succeeds
    # Read-only: assignment through the array interface errors without mutating.
    @test_throws Exception (w[1] = StageSolution(true))
    @test w[1].succeeds
    # A wrapper over a manually advanced (non-full) state only exposes solved stages.
    problem2 = SequentialProblem(
        StageProblem(true);
        transition = (sol, i) -> StageProblem(true),
        terminal = (sol, i) -> i >= 2,
    )
    it2 = init(problem2, NoSolve())
    append!(it2.buffer, [StageSolution(false), StageSolution(false)])
    w2 = solve!(it2)
    @test length(w2) == 2
    @test w2[1].succeeds && w2[2].succeeds
    @test_throws BoundsError w2[3]
    @test retcode(w2) == SciMLBase.ReturnCode.Success
end

@testset "No advancement after failure" begin
    attempts = Int[]
    problem = SequentialProblem(
        StageProblem(true);
        transition = (sol, i) -> begin
            push!(attempts, i)
            StageProblem(i != 2)
        end,
        terminal = (sol, i) -> i >= 5,
    )
    it = init(problem, NoSolve())
    w = solve!(it)
    @test attempts == [2]      # stage 2 fails; stage 3 is never attempted
    @test it.state == 2
    @test length(w) == 2
    @test retcode(w) == SciMLBase.ReturnCode.MaxIters
    # Re-solve! must not advance a failed state either.
    w2 = solve!(it)
    @test attempts == [2]
    @test it.state == 2
    @test length(w2) == 2
end
