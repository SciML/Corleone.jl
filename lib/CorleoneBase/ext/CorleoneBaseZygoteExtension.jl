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
import CorleoneBase: split_sequence_kwargs, get_problem, prob_func

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
#
# The differentiated replay must not go through SciMLBase's native ensemble
# adjoint (SciMLBaseZygoteExt's solve_batch pullback) -- that path assumes each
# trajectory solve returns a single state array, so a wrapped SequentialProblem
# trajectory (a Vector of per-stage solutions) breaks the cotangent broadcast
# ("array could not be broadcast to match destination"). It also must not build
# the SAME EnsembleSolution struct that Zygote then indexes: Zygote's
# getproperty-based struct adjoint errors on the wrapped Vector-of-solutions
# aggregation. Instead the replay returns the plain per-trajectory Vector
# (which Zygote aggregates correctly) and the rrule translates the ensemble
# solution's cotangent into that Vector cotangent before replaying.
function parallel_ensemble_serial_solve(problem, algorithm, kwargs)
    # kwargs is the full NamedTuple forwarded to CommonSolve.solve: split into
    # the required trajectories count and the solver kwargs. `_solver_kwargs`
    # reconstructs the solver kwargs as a *concrete* NamedTuple type and
    # `_solve_trajectory` re-emits each solver keyword individually in codegen,
    # so each per-trajectory solve stays type-stable (@inferred-clean, no
    # abstract-NamedTuple widening from Base.structdiff/select_keys).
    # CommonSolve.solve is itself a @generated kwcall: a runtime keyword list
    # defeats its output-type inference, which is precisely the instability this
    # construction removes. The results must be collected non-mutatingly so the
    # replay remains differentiable under Zygote (an elementwise setindex! loop
    # would trip Zygote's mutating-array check). Ensemble keys are not present
    # here: they configure the EnsembleProblem only and never reach the
    # per-trajectory solve.
    solver_kwargs = _solver_kwargs(kwargs)
    trajectories = kwargs.trajectories
    template = get_problem(problem)
    return map(1:trajectories) do i
        _solve_trajectory(problem, template, algorithm, solver_kwargs, i)
    end
end

# Reconstruct the solver-only kwargs (everything except :trajectories) as a
# concrete NamedTuple type. Base.structdiff cannot be proven concrete at
# inference (its result structure depends on runtime key sets), so this
# @generated helper builds the dropped NamedTuple directly from the concrete
# field names and drives the exact return type.
@generated function _solver_kwargs(kwargs::K) where {K <: NamedTuple}
    names = Tuple(filter(n -> n !== :trajectories, fieldnames(K)))
    reads = Expr(:tuple, (:(kwargs.$n) for n in names)...)
    return :(NamedTuple{$names}($reads))
end

# Solve one trajectory problem, re-emitting every solver keyword individually
# (Expr(:kw, ...)) so the @generated CommonSolve.solve kwcall sees a statically
# known keyword structure and infers a concrete ODESolution. `i` is the
# trajectory's ensemble sim_id, matching the real forward's `prob_func` context.
@generated function _solve_trajectory(problem, template, algorithm, sk::S, i) where {S <: NamedTuple}
    names = fieldnames(S)
    kws = [Expr(:kw, n, :(sk.$n)) for n in names]
    return quote
        ctx = _trajectory_context(i)
        trajectory = prob_func(problem, template, ctx)
        CommonSolve.solve(trajectory, algorithm; $(kws...))
    end
end

# Every trajectory context mirrors the ensemble context the real forward feeds
# `prob_func` (ctx.sim_id == 1:trajectories). Context creation carries no
# numerical derivative.
function _trajectory_context(i)
    return SciMLBase.EnsembleContext(
        i, 1, 0, nothing,
        SciMLBase.default_rng_func(
            SciMLBase.EnsembleContext(i, 1, 0, nothing, nothing, nothing)
        ), nothing
    )
end

# Pull the per-trajectory cotangent out of an EnsembleSolution-shaped cotangent.
# The loss only ever reads `ensemble.u`, so the `u` field carries the whole
# derivative. Handle AbstractZero, NamedTuple/struct tangents, and `AbstractArray`
# deltas uniformly via `unthunk`/`getproperty`. A wrapped trajectory's solution
# is a `Vector` of per-stage solutions, so the ensemble cotangent arrives as a
# dense `Matrix` keyed by (stage, trajectory); reshape it back into the nested
# `Vector{Vector{...}}` cotangent that the serial replay's output expects.
function _u_cotangent(delta)
    delta = unthunk(delta)
    delta isa AbstractZero && return delta
    if delta isa AbstractArray
        if ndims(delta) == 2
            # Wrapped trajectory solutions are Vectors of per-stage solutions, so
            # the ensemble cotangent arrives as a dense Matrix keyed by
            # (stage, trajectory): delta[k, i] is the cotangent of trajectory
            # i's k-th stage, i.e. of outputs[i][k]. Reshape it back into the
            # nested Vector{Vector{...}} cotangent the serial replay expects.
            return [[delta[k, i] for k in axes(delta, 1)] for i in axes(delta, 2)]
        end
        return hasproperty(delta, :u) ? delta.u : ZeroTangent()
    end
    return hasproperty(delta, :u) ? delta.u : ZeroTangent()
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
        ubar = _u_cotangent(delta)
        ubar isa AbstractZero && return (
            NoTangent(), ZeroTangent(), ZeroTangent(), NoTangent(),
        )
        gradients = back(ubar)
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
