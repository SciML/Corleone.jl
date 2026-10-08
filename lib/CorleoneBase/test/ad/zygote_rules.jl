module ZygoteRuleTests

using Test
using CorleoneBase
using CommonSolve
using SciMLBase
using Zygote

struct StageProblem{T}
    value::T
    succeeds::Bool
end

struct StageSolution{T}
    value::T
    succeeds::Bool
end

CommonSolve.solve(p::StageProblem, ::Nothing; kwargs...) = StageSolution(p.value, p.succeeds)
SciMLBase.successful_retcode(sol::StageSolution) = sol.succeeds

function iterator(x; terminal_at = 3, fail_at = 0, preallocate = false)
    problem = StageProblem(x[1], true)
    initial = solve(problem, nothing)
    buffer = preallocate ? vcat([initial], fill(StageSolution(0.0, true), 2)) : [initial]
    transition = (sol, i) -> StageProblem(sol.value * x[2], i != fail_at)
    terminal = (sol, i) -> i >= terminal_at
    return CorleoneBase.SequentialProblemIterator(
        problem, transition, terminal, nothing, (;), buffer, 1
    )
end

function loss(x; kwargs...)
    w = solve!(iterator(x; kwargs...))
    return sum(w[i].value for i in eachindex(w))
end

@testset "Zygote solve! rule contracts" begin
    @test Base.get_extension(CorleoneBase, :CorleoneBaseZygoteExtension) !== nothing
    x = [1.2, 0.4]
    for preallocate in (false, true), (terminal_at, fail_at, stages) in (
                (1, 0, 1), (2, 0, 2), (3, 0, 3), (3, 2, 2), (3, 3, 3),
            )
        options = (; terminal_at, fail_at, preallocate)
        it = iterator(x; options...)
        original_buffer = it.buffer
        result, back = Zygote.pullback(solve!, it)
        @test result isa CorleoneBase.SolutionWrapper
        @test result.buffer === original_buffer
        @test result.state == stages
        @test length(result.buffer) == (preallocate ? 3 : stages)
        @test back(nothing) === nothing

        value, gradient = Zygote.withgradient(x -> loss(x; options...), x)
        expected_value = sum(x[1] * x[2]^k for k in 0:(stages - 1))
        expected_gradient = [
            sum(x[2]^k for k in 0:(stages - 1)),
            sum(k * x[1] * x[2]^(k - 1) for k in 1:(stages - 1); init = 0.0),
        ]
        @test value ≈ expected_value
        @test only(gradient) ≈ expected_gradient
    end
    @test only(Zygote.gradient(x -> solve!(iterator(x))[1].value, x)) ≈ [1.0, 0.0]
end

end # module ZygoteRuleTests
