module CorleoneBaseZygoteExtension

using CorleoneBase
using CommonSolve
using SciMLBase
using ChainRulesCore
import Zygote
import CorleoneBase: SequentialProblemIterator, prepare_stage_problem
import CorleoneBase: AbstractSequentialProblem, get_problem, make_buffer, transition, terminal

# Repeated reads of a mutable iterator must return independent structural
# cotangents, not aliases of Zygote's mutable-object gradient cache.
function ChainRulesCore.rrule(::typeof(getproperty), it::SequentialProblemIterator, field::Symbol)
    value = getproperty(it, field)
    function property_pullback(delta)
        tangent = field === :state ? NoTangent() : unthunk(delta)
        return NoTangent(), Tangent{typeof(it)}(; field => tangent), NoTangent()
    end
    return value, property_pullback
end

initial_keyword_names(options) = Tuple(
    key for key in keys(options) if key in (:u0, :p, :tspan)
)
@non_differentiable initial_keyword_names(::Any)

function initial_stage(problem, algorithm, options)
    initial_kwargs = NamedTuple{initial_keyword_names(options)}(options)
    solve_kwargs = Base.structdiff(options, initial_kwargs)
    inner_problem = prepare_stage_problem(remake(get_problem(problem); initial_kwargs...))
    solution = solve(inner_problem, algorithm; solve_kwargs...)
    SciMLBase.successful_retcode(solution) || throw(ErrorException(
        "The initial call to solve failed with returncode $(solution.retcode)"
    ))
    return SequentialProblemIterator(
        inner_problem, Base.Fix1(transition, problem), Base.Fix1(terminal, problem),
        algorithm, solve_kwargs, make_buffer(problem, solution), 1
    )
end

Zygote.@adjoint function Core.kwcall(
    options::NamedTuple, ::typeof(CommonSolve.init),
    problem::AbstractSequentialProblem, algorithm
)
    it, back = Zygote.pullback(initial_stage, problem, algorithm, options)
    function init_pullback(delta)
        gradients = back(delta)
        gradients === nothing && return nothing
        problem_bar, algorithm_bar, options_bar = gradients
        return options_bar, nothing, problem_bar, algorithm_bar
    end
    return it, init_pullback
end

# Zygote's default mutable constructor also reads its identity-based gradient
# cache. A structural constructor rule avoids counting solve!'s tangent twice.
function ChainRulesCore.rrule(
    ::Type{SequentialProblemIterator}, problem, transition, terminal,
    algorithm, solve_kwargs, buffer, state
)
    it = SequentialProblemIterator(
        problem, transition, terminal, algorithm, solve_kwargs, buffer, state
    )
    function iterator_pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return (
            NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(),
            ZeroTangent(), ZeroTangent(), ZeroTangent(), NoTangent()
        )
        return (
            NoTangent(), delta.problem, delta.transition, delta.terminal,
            delta.algorithm, delta.solve_kwargs, delta.buffer, NoTangent()
        )
    end
    return it, iterator_pullback
end

# SciML's state access pullbacks use a DiffEqArray, whereas problem metadata
# uses a structural tangent. Buffer accumulation also requires matching keys.
function solution_cotangent(delta, solution)
    delta = unthunk(delta)
    state_array = delta isa AbstractArray && hasproperty(delta, :u)
    if solution isa SciMLBase.AbstractSciMLSolution &&
        (state_array || delta isa Tangent || delta isa NamedTuple)
        fields = fieldnames(typeof(solution))
        values = map(fields) do field
            if state_array
                field === :u ? delta.u : ZeroTangent()
            else
                hasproperty(delta, field) ? getproperty(delta, field) : ZeroTangent()
            end
        end
        return Tangent{Any}(; NamedTuple{fields}(values)...)
    end
    return delta
end

stage_solution(solution) = solution
function ChainRulesCore.rrule(::typeof(stage_solution), solution)
    return solution, delta -> (NoTangent(), solution_cotangent(delta, solution))
end

saved_buffer(buffer) = buffer
function ChainRulesCore.rrule(::typeof(saved_buffer), buffer)
    function buffer_pullback(delta)
        delta = unthunk(delta)
        return NoTangent(), delta isa AbstractZero ? delta : map(solution_cotangent, delta, buffer)
    end
    return buffer, buffer_pullback
end

# Buffer mutations have Zygote pullbacks. Initialize all existing entries and
# freeze only after the final stage, retaining any preallocated suffix.
function finish_stages(buffer, state, transition, terminal, algorithm, solve_kwargs)
    scratch = Zygote.Buffer(buffer)
    copyto!(scratch, buffer)
    while !terminal(stage_solution(scratch[state]), state)
        problem = prepare_stage_problem(transition(stage_solution(scratch[state]), state + 1))
        solution = solve(problem, algorithm; solve_kwargs...)
        SciMLBase.successful_retcode(solution) || break
        if length(scratch) >= state + 1
            scratch[state + 1] = solution
        else
            push!(scratch, solution)
        end
        state += 1
    end
    return saved_buffer(copy(scratch)), state
end

function ChainRulesCore.rrule(
    config::RuleConfig{>:HasReverseMode}, ::typeof(CommonSolve.solve!),
    it::SequentialProblemIterator
)
    (buffer, state), back = rrule_via_ad(
        config, finish_stages, copy(it.buffer), it.state,
        it.transition, it.terminal, it.algorithm, it.solve_kwargs
    )
    # Preserve solve!'s return identity and mutate the original buffer only
    # outside AD. The pullback owns the unmodified input buffer snapshot.
    for i in eachindex(buffer)
        if i <= length(it.buffer)
            it.buffer[i] = buffer[i]
        else
            push!(it.buffer, buffer[i])
        end
    end
    it.state = state
    function solve_pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent()
        _, buffer_bar, _, transition_bar, terminal_bar, algorithm_bar, kwargs_bar =
            back((unthunk(delta.buffer), NoTangent()))
        return NoTangent(), Tangent{typeof(it)}(;
            problem = unthunk(delta.problem),
            transition = add!!(transition_bar, unthunk(delta.transition)),
            terminal = add!!(terminal_bar, unthunk(delta.terminal)),
            algorithm = add!!(algorithm_bar, unthunk(delta.algorithm)),
            solve_kwargs = add!!(kwargs_bar, unthunk(delta.solve_kwargs)),
            buffer = buffer_bar, state = NoTangent()
        )
    end
    return it, solve_pullback
end

end # module CorleoneBaseZygoteExtension
