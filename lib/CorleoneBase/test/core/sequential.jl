using CorleoneBase
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
    @test solve!(it) === it
    @test it.buffer === buffer
    @test calls == [2, 3]
    @test Base.isdone(it)
    @test it.state == 3
    @test it.buffer[end].u[end][1] ≈ 1.2 * exp(-1.2)
    @test solve!(it) === it
    @test calls == [2, 3]

    single = solve(SequentialProblem(template), Tsit5(); SOLVE_KWARGS...)
    @test single.state == 1
    @test length(single.buffer) == 1
    @test Base.isdone(single)
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
SciMLBase.remake(p::StageProblem; kwargs...) = p
CommonSolve.solve(p::StageProblem, ::Nothing; kwargs...) = StageSolution(p.succeeds)
SciMLBase.successful_retcode(sol::StageSolution) = sol.succeeds

@testset "Return codes and preallocated buffers" begin
    @test_throws "The initial call to solve failed with returncode MaxIters" init(
        SequentialProblem(StageProblem(false)), nothing
    )
    for preallocate in (false, true), fail_at in (0, 2, 3)
        problem = SequentialProblem(
            StageProblem(true);
            transition = (sol, i) -> StageProblem(i != fail_at),
            terminal = (sol, i) -> i >= 3,
        )
        it = init(problem, nothing)
        if preallocate
            append!(it.buffer, [StageSolution(false), StageSolution(false)])
        end
        buffer = it.buffer
        @test solve!(it) === it
        @test it.buffer === buffer
        @test it.state == (fail_at == 0 ? 3 : fail_at - 1)
        @test length(it.buffer) == (preallocate ? 3 : it.state)
        @test all(sol.succeeds for sol in it.buffer[1:it.state])
    end
end
