using CorleoneBase
using CommonSolve: CommonSolve, init, solve, solve!, step!
using OrdinaryDiffEqTsit5: Tsit5
using SciMLBase: SciMLBase, ODEProblem, isinplace, remake, successful_retcode
using Test
import ArrayInterface

@test realpath(pkgdir(CorleoneBase)) == realpath(joinpath(@__DIR__, "../.."))
@test Base.get_extension(CorleoneBase, :CorleoneBaseZygoteExtension) === nothing
include("../ad/fixtures.jl")

@testset "Analytic multi-stage solutions" begin
    for case in CASES, x in INPUTS
        solutions = sequential_solutions(case, x)
        expected, _ = reference_endpoints(case, x)
        @test solutions isa Vector
        @test length(solutions) == 3
        for (i, sol) in enumerate(solutions)
            @test successful_retcode(sol)
            @test isinplace(sol.prob) == is_inplace(case)
            @test sol.prob.tspan == (Float64(i - 1), Float64(i))
            @test sol.t[end] == Float64(i)
            @test i == 1 ? sol.t[1] == 0.0 : sol.t[1] > Float64(i - 1)
            @test sol.prob.p == [x[i == 1 ? 2 : 3]]
            @test sol.prob.u0 == (i == 1 ? [x[1]] : i == 2 ? solutions[1].u[end] : [x[4]])
            @test isapprox(sol.u[end][1], expected[i]; atol = 1.0e-9, rtol = 1.0e-9)
        end
    end
end

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

# A concrete subtype exercises the abstract defaults and sized solve! dispatch.
struct SizedSequence <: CorleoneBase.AbstractSequentialProblem
    problem::StageProblem
    count::Int
    stop::Int
    fail_at::Int
end
CorleoneBase.get_problem(p::SizedSequence) = p.problem
CorleoneBase.get_size(p::SizedSequence) = p.count
CorleoneBase.has_size(::SizedSequence) = true
CorleoneBase.transition(p::SizedSequence, sol, i) = StageProblem(i != p.fail_at)
CorleoneBase.terminal(p::SizedSequence, sol, i) = i >= p.stop || !successful_retcode(sol)

struct FallbackSequence <: CorleoneBase.AbstractSequentialProblem
    problem::StageProblem
end
CorleoneBase.get_problem(p::FallbackSequence) = p.problem

@testset "Initialization, output, stepping, and buffer identity" begin
    template = ODEProblem(
        (u, p, t) -> -p .* u, [9.0], (0.0, 1.0), 4.0;
        abstol = 1.0e-11, reltol = 1.0e-11
    )
    calls = Int[]
    problem = SequentialProblem(
        template;
        transition = (seq, sol, i) -> begin
            push!(calls, i)
            i == 1 ? seq.problem : remake(sol.prob; u0 = sol.u[end], tspan = (sol.t[end], sol.t[end] + 1))
        end,
        terminal = (seq, sol, i) -> i >= 3
    )
    it = init(problem, Tsit5(); u0 = [1.2], p = 0.4, tspan = (2.0, 3.0))
    @test calls == [1]
    @test it.current == 1
    @test it.max_steps === nothing
    @test length(it.buffer) == 1
    @test !Base.isdone(it)
    @test it.problem.u0 == [1.2]
    @test it.problem.p == 0.4
    @test it.problem.tspan == (2.0, 3.0)
    @test template.u0 == [9.0] && template.p == 4.0 && template.tspan == (0.0, 1.0)
    @test it.output(it.buffer[1], 1) === it.buffer[1]
    buffer = it.buffer
    @test step!(it) === it
    @test it.current == 2
    @test calls == [1, 2]
    result = solve!(it)
    @test result !== buffer && buffer === it.buffer
    @test all(i -> result[i] === buffer[i], eachindex(result))
    @test calls == [1, 2, 3]
    @test Base.isdone(it)
    @test it.current == 3
    @test length(result) == 3
    @test result[3].u[end][1] ≈ 1.2 * exp(-1.2)
    @test all(successful_retcode, result)
    @test solve!(it) !== result
    @test calls == [1, 2, 3]

    outputs = Int[]
    seen = Int[]
    projected = SequentialProblem(
        StageProblem(true);
        transition = (seq, previous, i) -> begin
            i == 1 || push!(seen, previous)
            StageProblem(true)
        end,
        output = (seq, sol, i) -> (push!(outputs, i); i * 10),
        terminal = (seq, value, i) -> value >= 30
    )
    @test solve(projected, NoSolve()) == [10, 20, 30]
    @test outputs == [1, 2, 3]
    @test seen == [10, 20]
    @test CorleoneBase.output(FallbackSequence(StageProblem(true)), :value, 1) === :value
end

@testset "Terminal fallback, explicit failures, and preallocation" begin
    fallback = FallbackSequence(StageProblem(true))
    @test !CorleoneBase.has_size(fallback)
    @test CorleoneBase.get_size(fallback) === nothing
    @test CorleoneBase.transition(fallback, nothing, 1) === fallback.problem
    @test !CorleoneBase.terminal(fallback, StageSolution(true), 0)
    @test CorleoneBase.terminal(fallback, StageSolution(false), 0)
    @test CorleoneBase.terminal(fallback, StageSolution(true), 1)
    @test length(solve(fallback, NoSolve())) == 1
    for sized in (false, true), stop in (1, 2, 3), fail_at in (0, 1, 2, 3)
        problem = sized ? SizedSequence(StageProblem(true), 5, stop, fail_at) :
            SequentialProblem(
                StageProblem(true);
                transition = (seq, sol, i) -> StageProblem(i != fail_at),
                terminal = (seq, sol, i) -> i >= stop || !successful_retcode(sol)
            )
        it = init(problem, NoSolve())
        expected = fail_at == 0 ? stop : min(stop, fail_at)
        @test it.max_steps === (sized ? 5 : nothing)
        @test length(it.buffer) == (sized ? 5 : 1)
        buffer = it.buffer
        suffix_assigned = [isassigned(buffer, i) for i in (expected + 1):length(buffer)]
        result = solve!(it)
        @test result !== buffer && it.buffer === buffer
        @test it.current == expected
        @test length(result) == expected
        @test length(buffer) == (sized ? 5 : expected)
        @test successful_retcode(result[it.current]) == (fail_at == 0 || fail_at > stop)
        @test all(i -> isassigned(result, i), 1:expected)
        @test CorleoneBase.finalize_buffer(it) !== buffer
        @test solve!(it) !== result
        @test all(i -> result[i] === buffer[i], eachindex(result))
        if sized
            # Inline/isbits elements are always reported assigned, even in an
            # unwritten slot. Finalization must preserve the suffix either way.
            @test [isassigned(buffer, i) for i in (expected + 1):length(buffer)] == suffix_assigned
        end
        @test_throws BoundsError result[expected + 1]
    end
    # Custom terminals own failure handling: no implicit success check.
    explicit = SequentialProblem(
        StageProblem(false);
        transition = (seq, sol, i) -> StageProblem(false), terminal = (seq, sol, i) -> i >= 3
    )
    @test length(solve(explicit, NoSolve())) == 3
    # Final buffers are ordinary mutable vectors, not read-only views.
    result = solve(fallback, NoSolve())
    result[1] = StageSolution(false)
    @test !successful_retcode(result[1])
end

# Capture both remake options and solve flags without accepting them silently.
struct KwProblem <: SciMLBase.AbstractSciMLProblem
    tag::Symbol
    options::NamedTuple
end
KwProblem(tag) = KwProblem(tag, (;))
SciMLBase.remake(p::KwProblem; tag = p.tag, options = p.options, kwargs...) =
    KwProblem(tag, merge(options, NamedTuple(kwargs)))
struct KwSolution
    prob::KwProblem
    flags::NamedTuple
end
CommonSolve.solve(p::KwProblem, ::NoSolve; kwargs...) = KwSolution(p, NamedTuple(kwargs))

struct SizedKwSequence <: CorleoneBase.AbstractSequentialProblem end
CorleoneBase.has_size(::SizedKwSequence) = true
CorleoneBase.get_size(::SizedKwSequence) = 4
CorleoneBase.transition(::SizedKwSequence, sol, i) = KwProblem(Symbol("stage", i))
CorleoneBase.terminal(::SizedKwSequence, sol, i) = i >= 3

@testset "Sized solve! resumes the current stage after step!" begin
    it = init(SizedKwSequence(), NoSolve())
    @test step!(it) === it
    @test it.current == 2
    buffer = it.buffer
    result = solve!(it)
    @test result !== buffer && it.buffer === buffer
    @test it.current == 3
    @test length(result) == 3
    @test length(buffer) == it.max_steps == 4
    @test !isassigned(buffer, 4)
    @test [result[i].prob.tag for i in 1:it.current] == [:stage1, :stage2, :stage3]
    sentinel = KwSolution(KwProblem(:unused), (;))
    buffer[4] = sentinel
    @test CorleoneBase.finalize_buffer(it) !== buffer
    @test buffer[4] === sentinel
    @test solve!(it) !== result
    @test buffer[4] === sentinel
    result[1] = sentinel
    @test buffer[1].prob.tag == :stage1
end

@testset "Sequence routing and first-stage remake" begin
    @test CorleoneBase.AbstractSequentialProblem <: SciMLBase.AbstractSciMLProblem
    calls = []
    template = KwProblem(:template)
    problem = SequentialProblem(
        template;
        transition = (seq, sol, i) -> begin
            push!(calls, (i, sol, seq.parameters, seq.states))
            KwProblem(i == 1 ? :constructed : :later)
        end,
        terminal = (seq, sol, i) -> i >= 2
    )
    @test_throws MethodError init(problem, nothing)
    @test_throws MethodError init(problem, :not_an_algorithm)
    @test problem.parameters === problem.states === nothing
    it = init(problem, NoSolve(); tag = :override, extra = 2, parameters = :parameters, states = :states)
    @test calls == [(1, nothing, :parameters, :states)]
    @test it.problem.tag == :override
    @test it.problem.options == (; extra = 2)
    @test it.buffer[1].flags == (; dense = false, save_everystep = false, save_start = true, save_end = true)
    result = solve!(it)
    @test result[2].prob.tag == :later
    @test isempty(result[2].prob.options)
    @test result[2].flags == (; save_start = false, save_end = true)
    @test calls[2][1] == 2 && calls[2][2] === result[1]
    @test template.tag == :template
    @test problem.parameters === problem.states === nothing
    projection = (seq, sol, i) -> sol.prob.tag
    remade = remake(problem; output = projection, parameters = :new, states = [1.0])
    @test remade !== problem
    @test remade.output === projection
    @test remade.parameters == :new && remade.states == [1.0]
    @test remade.problem === problem.problem
    @test remade.transition === problem.transition
    @test remade.terminal === problem.terminal
    seq_options, rest = CorleoneBase.split_sequence_kwargs(
        problem,
        (; output = projection, parameters = 1, states = 2, tag = :a, extra = 3)
    )
    @test seq_options == (; output = projection, parameters = 1, states = 2)
    @test rest == (; tag = :a, extra = 3)
    @test CorleoneBase.split_sequence_kwargs(problem, (;)) == ((;), (;))
    @test solve(problem, NoSolve(); output = projection) == [:constructed, :later]
end

@testset "Callbacks require the sequence argument" begin
    legacy = SequentialProblem(StageProblem(true); transition = (sol, i) -> StageProblem(true))
    @test_throws MethodError init(legacy, NoSolve())
    legacy_terminal = SequentialProblem(
        StageProblem(true);
        transition = (seq, sol, i) -> seq.problem, terminal = (sol, i) -> true
    )
    @test_throws MethodError Base.isdone(init(legacy_terminal, NoSolve()))
    legacy_output = SequentialProblem(
        StageProblem(true);
        transition = (seq, sol, i) -> seq.problem, output = (sol, i) -> sol
    )
    @test_throws MethodError init(legacy_output, NoSolve())
end

struct AoSMarker end
struct SoAMarker end
struct MarkerProblem <: SciMLBase.AbstractSciMLProblem
    data
end
ArrayInterface.aos_to_soa(::AoSMarker) = SoAMarker()
SciMLBase.remake(p::MarkerProblem; data = p.data) = MarkerProblem(data)
struct AoSVector <: AbstractVector{Float64}
    n::Int
end
Base.size(v::AoSVector) = (v.n,)
Base.getindex(::AoSVector, i::Int) = 0.0
ArrayInterface.aos_to_soa(v::AoSVector) = zeros(Float64, v.n)

@testset "Stage preparation" begin
    @test CorleoneBase.prepare_stage_problem(MarkerProblem(AoSMarker())).data isa SoAMarker
    @test CorleoneBase.prepare_stage_problem(:non_sciml) === :non_sciml
    converting = ODEProblem((u, p, t) -> -p .* u, AoSVector(1), (0.0, 1.0), [2.0])
    prepared = CorleoneBase.prepare_stage_problem(converting)
    @test prepared !== converting
    @test prepared.u0 isa Vector{Float64}
    @test prepared.p === converting.p
    @test prepared.tspan == converting.tspan
    ode = ODEProblem((u, p, t) -> -p .* u, [1.0], (0.0, 1.0), [2.0])
    @test CorleoneBase.prepare_stage_problem(ode) === ode
    inplace = ODEProblem((du, u, p, t) -> (du .= -p .* u; nothing), [1.0], (0.0, 1.0), [2.0])
    @test isinplace(inplace)
    @test CorleoneBase.prepare_stage_problem(inplace) === inplace
end
