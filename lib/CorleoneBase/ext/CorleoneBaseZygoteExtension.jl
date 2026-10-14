module CorleoneBaseZygoteExtension

using CommonSolve: CommonSolve, solve
using SciMLBase: SciMLBase, remake
using ChainRulesCore: ChainRulesCore, AbstractZero, HasReverseMode, NoTangent,
    RuleConfig, Tangent, ZeroTangent, rrule_via_ad, unthunk
import Zygote
import CorleoneBase: SequentialProblemIterator, SolutionWrapper, prepare_stage_problem
import CorleoneBase: AbstractSequentialProblem, get_problem, make_buffer, transition, terminal
import CorleoneBase: AbstractParallelProblem
# Shared EnsembleProblem/keyword helpers defined by the parent package; they are
# implementation detail shared with the extensions, not public API.
import CorleoneBase: ENSEMBLE_KEYS, ensemble_problem, select_keys, split_problem_kwargs,
    split_sequence_kwargs

# The parallel ensemble fans out across a SciML EnsembleProblem and, under
# EnsembleThreads or EnsembleDistributed, across threads or worker processes.
# Zygote's reverse pass is not thread-safe (the wrapped SequentialProblem
# trajectories nondeterministically returned wrong gradients under
# EnsembleThreads), and its shadow/pullback structures cannot survive
# Distributed's pmap serialization (EnsembleDistributed workers crashed). Solve
# the real ensemble as an opaque forward and re-differentiate the actual
# trajectory problems serially in the pullback. The forward genuinely executes
# on threads/workers (so execution-location probes hold), ensemble solutions do
# not depend on the mode, and the adjoint recomputes every trajectory's true
# trajectory-specific gradient without racing or serializing a Zygote tape.
function parallel_ensemble_serial_solve(problem, algorithm, kwargs)
    # kwargs is the full NamedTuple forwarded to CommonSolve.solve: split into
    # EnsembleProblem options, the required trajectories count, and solver kwargs.
    ensemble_kwargs = select_keys(kwargs, ENSEMBLE_KEYS)
    solve_kwargs = Base.structdiff(kwargs, ensemble_kwargs)
    trajectories = solve_kwargs.trajectories
    solver_kwargs = Base.structdiff(solve_kwargs, (; trajectories = nothing))
    replay_problem = ensemble_problem(problem; ensemble_kwargs...)
    return SciMLBase.solve(
        replay_problem, algorithm, SciMLBase.EnsembleSerial();
        trajectories = trajectories, solver_kwargs...
    )
end

function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(CommonSolve.solve),
        problem::AbstractParallelProblem, algorithm, ensemblealg;
        kwargs...
    )
    nt = NamedTuple(kwargs)
    # Opaque forward: the real ensemble solve under the requested mode. Because
    # this body runs outside Zygote's recorder, threaded/distributed execution
    # neither races Zygote's reverse pass nor serializes a tape to workers.
    ensemble = CommonSolve.solve(problem, algorithm, ensemblealg; nt...)
    _, back = rrule_via_ad(
        config, parallel_ensemble_serial_solve, problem, algorithm, nt
    )
    function ensemble_pullback(delta)
        gradients = back(delta)
        gradients === nothing && return (
            NoTangent(), NoTangent(), NoTangent(), NoTangent(),
        )
        _, problem_bar, algorithm_bar, _ = gradients
        return NoTangent(), problem_bar, algorithm_bar, NoTangent()
    end
    return ensemble, ensemble_pullback
end

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

function initial_stage(problem, algorithm, options)
    # Route by field names in two steps, exactly like the primal
    # CommonSolve.init does: keywords naming a field of the sequence remake the
    # sequence the callbacks observe, keywords naming a field of the wrapped
    # problem (for an ODEProblem, the non-(u0,p,tspan) fields `f`, `kwargs`, and
    # `problem_type` included) are remade into the first stage, and the
    # remainder reach every stage's solve.
    seq_kwargs, options = split_sequence_kwargs(problem, options)
    sequence = isempty(seq_kwargs) ? problem : remake(problem; seq_kwargs...)
    template = get_problem(sequence)
    initial_kwargs, solve_kwargs = split_problem_kwargs(template, options)
    inner_problem = prepare_stage_problem(remake(template; initial_kwargs...))
    solution = solve(inner_problem, algorithm; solve_kwargs...)
    return SequentialProblemIterator(
        inner_problem, (sol, i) -> transition(sequence, sol, i), (sol, i) -> terminal(sequence, sol, i),
        algorithm, solve_kwargs, make_buffer(sequence, solution), 1
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
            ZeroTangent(), ZeroTangent(), ZeroTangent(), NoTangent(),
        )
        return (
            NoTangent(), delta.problem, delta.transition, delta.terminal,
            delta.algorithm, delta.solve_kwargs, delta.buffer, NoTangent(),
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

# Buffer mutations have Zygote pullbacks. Retain every attempted stage, including
# the first unsuccessful one, and stop without transitioning or solving further.
# Only solved stages are returned, so preallocated slots never leak into the
# returned wrapper's gradient space.
function finish_stages(buffer, state, transition, terminal, algorithm, solve_kwargs)
    scratch = Zygote.Buffer(buffer)
    copyto!(scratch, buffer)
    while SciMLBase.successful_retcode(stage_solution(scratch[state])) &&
            !terminal(stage_solution(scratch[state]), state)
        problem = prepare_stage_problem(transition(stage_solution(scratch[state]), state + 1))
        solution = solve(problem, algorithm; solve_kwargs...)
        if length(scratch) >= state + 1
            scratch[state + 1] = solution
        else
            push!(scratch, solution)
        end
        state += 1
        SciMLBase.successful_retcode(solution) || break
    end
    return saved_buffer(copy(scratch)[1:state]), state
end

function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(CommonSolve.solve!),
        it::SequentialProblemIterator
    )
    (buffer, state), back = rrule_via_ad(
        config, finish_stages, copy(it.buffer), it.state,
        it.transition, it.terminal, it.algorithm, it.solve_kwargs
    )
    # Preserve solve!'s buffer identity: write the finished solved stages back
    # into the original buffer, leaving any preallocated suffix untouched.
    for i in 1:state
        if i <= length(it.buffer)
            it.buffer[i] = buffer[i]
        else
            push!(it.buffer, buffer[i])
        end
    end
    it.state = state
    wrapper = SolutionWrapper(it.buffer, it.state)
    function solve_pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent()
        buffer_bar = unthunk(getproperty(delta, :buffer))
        _, buffer_bar_out, _, transition_bar, terminal_bar, algorithm_bar, kwargs_bar =
            back((buffer_bar, NoTangent()))
        # The wrapper exposes only its buffer, so downstream gradients reach the
        # iterator solely through the buffer; the other iterator fields pick up
        # their gradients from `finish_stages`'s args instead of delta.
        return NoTangent(), Tangent{typeof(it)}(;
                problem = ZeroTangent(),
                transition = transition_bar,
                terminal = terminal_bar,
                algorithm = algorithm_bar,
                solve_kwargs = kwargs_bar,
                buffer = buffer_bar_out, state = NoTangent()
            )
    end
    return wrapper, solve_pullback
end

# Read-only indexing into a SolutionWrapper: the underlying buffer is read, and
# the buffer cotangent is accumulated in the wrapper's `1:length` index space.
# Each element's cotangent is a solution-struct tangent, so the buffer cotangent
# is an untyped vector holding those per-slot tangents.
function ChainRulesCore.rrule(::typeof(getindex), w::SolutionWrapper, i::Int)
    value = w[i]
    function wrapper_index_pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent(), NoTangent()
        buffer_cot = Vector{Any}(undef, length(w))
        fill!(buffer_cot, ZeroTangent())
        buffer_cot[i] = delta
        return NoTangent(),
            Tangent{typeof(w)}(buffer = buffer_cot, state = NoTangent()), NoTangent()
    end
    return value, wrapper_index_pullback
end

end # module CorleoneBaseZygoteExtension
