module CorleoneBaseZygoteExtension

using CommonSolve: CommonSolve, solve
using SciMLBase: SciMLBase, remake
using ChainRulesCore: ChainRulesCore, AbstractZero, HasReverseMode, NoTangent,
    RuleConfig, Tangent, ZeroTangent, rrule_via_ad, unthunk
import Zygote
import CorleoneBase: SequentialProblemIterator, prepare_stage_problem
import CorleoneBase: AbstractSequentialProblem, make_buffer, transition, output, terminal,
    has_size, get_size
import CorleoneBase: AbstractParallelProblem
# Shared EnsembleProblem/keyword helpers defined by the parent package; they are
# implementation detail shared with the extensions, not public API.
import CorleoneBase: ENSEMBLE_KEYS, ensemble_problem, select_keys,
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
        tangent = field in (:current, :max_steps) ? NoTangent() : unthunk(delta)
        return NoTangent(), Tangent{typeof(it)}(; field => tangent), NoTangent()
    end
    return value, property_pullback
end

function initial_stage(problem, algorithm, options)
    # Match the primal: all non-sequence options remake the constructed stage.
    seq_kwargs, options = split_sequence_kwargs(problem, options)
    sequence = isempty(seq_kwargs) ? problem : remake(problem; seq_kwargs...)
    inner_problem = prepare_stage_problem(remake(transition(sequence, nothing, 1); options...))
    solution = solve(
        inner_problem, algorithm;
        dense = false, save_everystep = false, save_start = true, save_end = true
    )
    first_output = output(sequence, solution, 1)
    return SequentialProblemIterator(
        inner_problem, (sol, i) -> transition(sequence, sol, i),
        (sol, i) -> output(sequence, sol, i), (sol, i) -> terminal(sequence, sol, i),
        algorithm, make_buffer(sequence, first_output), 1,
        has_size(sequence) ? get_size(sequence) : nothing
    )
end

function ChainRulesCore.rrule(::typeof(make_buffer), sequence::AbstractSequentialProblem, first_output)
    buffer = make_buffer(sequence, first_output)
    function make_buffer_pullback(delta)
        delta = unthunk(delta)
        return NoTangent(), NoTangent(), delta isa AbstractZero ? delta : delta[1]
    end
    return buffer, make_buffer_pullback
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
        ::Type{SequentialProblemIterator}, problem, transition, output, terminal,
        algorithm, buffer, current, max_steps
    )
    it = SequentialProblemIterator(
        problem, transition, output, terminal, algorithm, buffer, current, max_steps
    )
    function iterator_pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return (
            NoTangent(), ZeroTangent(), ZeroTangent(), ZeroTangent(),
            ZeroTangent(), ZeroTangent(), ZeroTangent(), NoTangent(), NoTangent(),
        )
        return (
            NoTangent(), delta.problem, delta.transition, delta.output, delta.terminal,
            delta.algorithm, delta.buffer, NoTangent(), NoTangent(),
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

# Buffer mutations have Zygote pullbacks. Only the terminal controls stopping,
# and every stage is transformed before the next transition sees it.
function finish_stages(buffer, current, transition, output, terminal, algorithm, max_steps)
    scratch = Zygote.Buffer(buffer)
    copyto!(scratch, buffer)
    while !terminal(stage_solution(scratch[current]), current)
        next = current + 1
        isnothing(max_steps) || next <= max_steps || throw(BoundsError(buffer, next))
        problem = prepare_stage_problem(transition(stage_solution(scratch[current]), next))
        solution = solve(problem, algorithm; save_start = false, save_end = true)
        push!(scratch, output(solution, next))
        current = next
    end
    return saved_buffer(copy(scratch)), current
end

function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode}, ::typeof(CommonSolve.solve!),
        it::SequentialProblemIterator
    )
    original_length = length(it.buffer)
    (buffer, current), back = rrule_via_ad(
        config, finish_stages, it.buffer[1:it.current], it.current,
        it.transition, it.output, it.terminal, it.algorithm, it.max_steps
    )
    # Update the iterator while preserving its capacity and untouched suffix.
    # Return the separate solved-prefix vector, exactly like finalize_buffer.
    for i in 1:current
        if i <= length(it.buffer)
            it.buffer[i] = buffer[i]
        else
            push!(it.buffer, buffer[i])
        end
    end
    it.current = current
    function solve_pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent()
        _, buffer_bar_out, _, transition_bar, output_bar, terminal_bar, algorithm_bar, _ =
            back((delta, NoTangent()))
        # Restore the input buffer's cotangent index space without reading its
        # uninitialized preallocated suffix.
        padded = buffer_bar_out isa AbstractZero ?
            fill(ZeroTangent(), original_length) :
            vcat(buffer_bar_out, fill(ZeroTangent(), original_length - length(buffer_bar_out)))
        return NoTangent(), Tangent{typeof(it)}(;
                problem = ZeroTangent(),
                transition = transition_bar,
                output = output_bar,
                terminal = terminal_bar,
                algorithm = algorithm_bar,
                buffer = padded, current = NoTangent(), max_steps = NoTangent()
            )
    end
    return buffer, solve_pullback
end

end # module CorleoneBaseZygoteExtension
