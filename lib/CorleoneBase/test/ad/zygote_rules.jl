module ZygoteRuleTests

using Test
using CorleoneBase
using CommonSolve
using SciMLBase
using SciMLBase: successful_retcode
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

# Every non-sequence keyword must reach the constructed stage's remake.
struct KwProblem{V} <: SciMLBase.AbstractSciMLProblem
    value::V
    tag::Symbol
    retcode::SciMLBase.ReturnCode.T
end
KwProblem(value, tag) = KwProblem(value, tag, SciMLBase.ReturnCode.Success)
SciMLBase.remake(p::KwProblem; value = p.value, tag = p.tag, extra = nothing) = KwProblem(value, tag)

struct NoSolve <: SciMLBase.AbstractSciMLAlgorithm end
CommonSolve.solve(p::KwProblem, ::NoSolve; kwargs...) = p
SciMLBase.successful_retcode(p::KwProblem) = true

function iterator(x; terminal_at = 3, fail_at = 0, preallocate = false)
    problem = StageProblem(x[1], true)
    initial = solve(problem, nothing)
    buffer = preallocate ? vcat([initial], fill(StageSolution(x[1]^2, true), 2)) : [initial]
    transition = (sol, i) -> StageProblem(sol.value * x[2], i != fail_at)
    terminal = (sol, i) -> i >= terminal_at || !successful_retcode(sol)
    return CorleoneBase.SequentialProblemIterator(
        problem, transition, (sol, i) -> sol, terminal, nothing, buffer, 1,
        preallocate ? 3 : nothing
    )
end

function loss(x; kwargs...)
    it = iterator(x; kwargs...)
    w = solve!(it)
    return sum(w[i].value for i in 1:it.current)
end

@testset "Zygote init keyword routing matches the primal" begin
    # Both field and non-field options reach remake; value retains its gradient.
    primal_loss(x) = solve!(
        init(
            SequentialProblem(KwProblem(0.5, :a);
                transition = (seq, sol, i) -> seq.problem, terminal = (seq, sol, i) -> true),
            NoSolve();
            value = x[1], tag = :b, extra = 1
        )
    )[1].value
    @testset "routing" begin
        template = SequentialProblem(KwProblem(0.5, :a);
            transition = (seq, sol, i) -> seq.problem, terminal = (seq, sol, i) -> true)
        it = init(template, NoSolve(); value = 1.3, tag = :b, extra = 1)
        @test it.problem isa KwProblem
        @test it.problem.tag == :b
        @test it.problem.value == 1.3
        @test it.current == 1 && it.max_steps === nothing
    end
    x = [1.7]
    primal_value = primal_loss(x)
    zygote_value, zygote_gradient = Zygote.withgradient(primal_loss, x)
    @test zygote_value == primal_value
    @test only(zygote_gradient) ≈ [1.0]
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
        @test result isa Vector
        @test result !== original_buffer && it.buffer === original_buffer
        @test it.current == stages
        @test length(result) == stages
        @test length(it.buffer) == (preallocate ? 3 : stages)
        if preallocate
            @test all(i -> it.buffer[i].value == x[1]^2, (stages + 1):length(it.buffer))
        end
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
    @test_throws BoundsError solve!(iterator(x; terminal_at = 1, preallocate = true))[3]
end

struct SizedRuleSequence{T} <: CorleoneBase.AbstractSequentialProblem
    value::T
end
CorleoneBase.has_size(::SizedRuleSequence) = true
CorleoneBase.get_size(::SizedRuleSequence) = 4
CorleoneBase.transition(seq::SizedRuleSequence, previous, i) =
    KwProblem(i == 1 ? seq.value : previous.value * seq.value, :stage)
CorleoneBase.output(seq::SizedRuleSequence, sol, i) = KwProblem(2 * sol.value, sol.tag)
CorleoneBase.terminal(::SizedRuleSequence, sol, i) = i >= 3

@testset "Zygote sized initialization and differentiated output" begin
    objective(x) = sum(sol.value for sol in solve(SizedRuleSequence(x[1]), NoSolve()))
    x = [0.4]
    value, gradient = Zygote.withgradient(objective, x)
    @test value ≈ 2x[1] + 4x[1]^2 + 8x[1]^3
    @test only(gradient) ≈ [2 + 8x[1] + 24x[1]^2]
    it = init(SizedRuleSequence(x[1]), NoSolve())
    @test length(it.buffer) == it.max_steps == 4
    result, back = Zygote.pullback(solve!, it)
    @test result !== it.buffer
    @test length(result) == 3
    @test length(it.buffer) == it.max_steps == 4
    @test it.current == 3
    @test !isassigned(it.buffer, 4)
    @test back(nothing) === nothing
end

end # module ZygoteRuleTests
