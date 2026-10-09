"""
    AbstractSequentialProblem <: SciMLBase.AbstractSciMLProblem

Supertype for problems solved stage by stage by [`SequentialProblem`](@ref).
Concrete subtypes provide `get_problem`, `transition`, and `terminal`. Because
every sequential problem is itself a `SciMLBase.AbstractSciMLProblem`, a
sequential problem can be used as the template of an outer solve or ensemble
(for example inside a [`ParallelProblem`](@ref)).
"""
abstract type AbstractSequentialProblem <: SciMLBase.AbstractSciMLProblem end

"""
    get_problem(problem::AbstractSequentialProblem)

Return the initial SciML problem solved as stage 1. Implemented by concrete
sequential problems; see [`SequentialProblem`](@ref).
"""
function get_problem(::AbstractSequentialProblem) end

"""
    transition(problem::AbstractSequentialProblem, sol, i)

Return the problem for next stage `i` (`i >= 2`) given the preceding stage's
solution `sol`. Implemented by concrete sequential problems; see
[`SequentialProblem`](@ref).
"""
function transition(::AbstractSequentialProblem, sol, i) end

"""
    terminal(problem::AbstractSequentialProblem, sol, i)

Return `true` when stage `i`'s solution `sol` ends the sequence (`i` starts at
1). Implemented by concrete sequential problems; see
[`SequentialProblem`](@ref).
"""
function terminal(::AbstractSequentialProblem, sol, i) end


"""
    make_buffer(problem::AbstractSequentialProblem, sol::T) -> Vector{T}

Allocate the stage-solution buffer seeded with the first stage's solution `sol`.
The element type is `T`, the concrete type of `sol`, so every later stage must
produce that exact type.
"""
function make_buffer(::AbstractSequentialProblem, sol::T) where {T}
    return T[sol]
end


"""
    SequentialProblemIterator

Mutable CommonSolve state returned by `CommonSolve.init(::SequentialProblem, alg)`.
It stores the first stage's remade `problem`, the `transition`/`terminal`
callbacks bound to the parent problem, the `algorithm`, the `solve_kwargs`
forwarded to every stage, the stage-solution `buffer`, and the current 1-based
`state`. `CommonSolve.step!` solves one more stage; `CommonSolve.solve!` runs
until the terminal predicate fires or a stage fails and returns a
[`SolutionWrapper`](@ref).
"""
mutable struct SequentialProblemIterator{P, N, T, A, K, B, S}
    const problem::P
    const transition::N
    const terminal::T
    const algorithm::A
    const solve_kwargs::K
    const buffer::B
    state::S
end


# 1. The generated function does the tuple intersection at compile-time
"""
    _split_problem_kwargs(problem, nt_kwargs::NamedTuple) -> (prob_kwargs, solve_kwargs)

Generated implementation of [`split_problem_kwargs`](@ref). The partition is
resolved at compile time from `fieldnames(typeof(problem))`, so the returned
`NamedTuple`s keep concrete key sets; `ArrayInterface.aos_to_soa` is applied to
each value routed to `prob_kwargs`.
"""
@generated function _split_problem_kwargs(::P, nt_kwargs::NamedTuple{K}) where {P <: SciMLBase.AbstractSciMLProblem, K}
    # These execute during compilation:
    valid_keys = fieldnames(P)
    prob_keys = Tuple(k for k in K if k in valid_keys)
    solve_keys = Tuple(k for k in K if !(k in valid_keys))

    # This generates the exact type-stable slicing code for the specific kwargs passed
    return quote
        prob_kwargs = map(ArrayInterface.aos_to_soa, NamedTuple{$prob_keys}(nt_kwargs))
        solve_kwargs = NamedTuple{$solve_keys}(nt_kwargs)
        return prob_kwargs, solve_kwargs
    end
end

"""
    _prepare_problem(problem) -> problem′

Generated helper for [`prepare_stage_problem`](@ref): rebuild `problem` through
`SciMLBase.remake` with `ArrayInterface.aos_to_soa` applied to every field.
"""
@generated function _prepare_problem(x::P) where {P}
    f_names = fieldnames(P)
    kw_exprs = [
        Expr(:kw, f, :(ArrayInterface.aos_to_soa(getfield(x, $(QuoteNode(f))))))
            for f in f_names
    ]
    return Expr(:call, :remake, Expr(:parameters, kw_exprs...), :x)
end

"""
    prepare_stage_problem(problem::SciMLBase.AbstractSciMLProblem) -> problem′

Return `problem` rebuilt through `SciMLBase.remake` with
`ArrayInterface.aos_to_soa` applied to every field. Out-of-place solvers require
the state and derivative to share one array representation; AD broadcasts may
leave a struct-of-arrays representation, which this normalizes before solving.
Fields that are already contiguous (plain vectors, scalars, functions) are
returned unchanged. Unlike the earlier ODE-only special case, this method
accepts any `SciMLBase.AbstractSciMLProblem`, and it always remakes rather than
returning the argument when no field changed.
"""
@inline prepare_stage_problem(x::SciMLBase.AbstractSciMLProblem) = _prepare_problem(x)

# 2. Your frontend function simply converts the kwargs to a NamedTuple and forwards it
"""
    split_problem_kwargs(problem, kwargs) -> (prob_kwargs, solve_kwargs)

Partition a keyword collection for `CommonSolve.init` using `problem`'s field
names. Keywords whose name is a field of `typeof(problem)` become `prob_kwargs`,
with `ArrayInterface.aos_to_soa` applied to their values, and are passed to
`SciMLBase.remake`; every other keyword becomes `solve_kwargs` and is forwarded
unchanged to each stage's solve. For an `ODEProblem` the problem fields are
`f`, `u0`, `tspan`, `p`, `kwargs`, and `problem_type`. The partition is decided
at compile time.
"""
@inline function split_problem_kwargs(prob::SciMLBase.AbstractSciMLProblem, kwargs)
    return _split_problem_kwargs(prob, NamedTuple(kwargs))
end

"""
    init(problem::SciMLBase.AbstractSciMLProblem, algorithm::SciMLBase.AbstractSciMLAlgorithm; kwargs...)

Initialize `problem` and immediately solve its first stage.

Keywords whose name is a field of the wrapped SciML problem (for an
`ODEProblem`: `u0`, `p`, `tspan`, `f`, `kwargs`, `problem_type`) are passed to
`SciMLBase.remake` to build the first stage, with `ArrayInterface.aos_to_soa`
applied to their values. All remaining keywords are stored and forwarded
unchanged to every stage's solve. Later stages come from the transition; the
initial overrides are not reapplied.

Returns a [`SequentialProblemIterator`](@ref) whose `buffer` already holds the
first stage's solution and whose `state` is `1`. The `algorithm` must be a
`SciMLBase.AbstractSciMLAlgorithm`; arbitrary solver-state objects and `nothing`
are rejected by dispatch.
"""
@inline function CommonSolve.init(
        problem::T, algorithm::SciMLBase.AbstractSciMLAlgorithm;
        kwargs...
    ) where {T <: SciMLBase.AbstractSciMLProblem}

    prob = get_problem(problem)
    initial_kwargs, solve_kwargs = split_problem_kwargs(prob, kwargs)
    inner_problem = remake(prob; initial_kwargs...)
    sol = solve(inner_problem, algorithm; solve_kwargs...)
    buffer = make_buffer(problem, sol)
    # Fix1 accepts only one remaining argument on Julia 1.10/1.11.
    return SequentialProblemIterator(
        inner_problem,
        (sol, i) -> transition(problem, sol, i),
        (sol, i) -> terminal(problem, sol, i),
        algorithm,
        solve_kwargs,
        buffer,
        1
    )
end

function increment!(it::SequentialProblemIterator)
    it.state += 1
    return it
end

function CommonSolve.step!(it::SequentialProblemIterator)
    (; algorithm, solve_kwargs, transition, buffer, state) = it
    current_problem = prepare_stage_problem(transition(buffer[state], state + 1))
    current_solution = solve(current_problem, algorithm; solve_kwargs...)
    if length(buffer) >= state + 1
        buffer[state + 1] = current_solution
    else
        push!(buffer, current_solution)
    end
    increment!(it)
    SciMLBase.successful_retcode(current_solution) || return false
    return true
end

function Base.isdone(it::SequentialProblemIterator, state = it.state)
    return it.terminal(it.buffer[state], state)
end


function CommonSolve.solve!(it::SequentialProblemIterator)
    # Gate the loop on the last retained stage's success so an unsuccessful first
    # stage (already in the buffer from init) never transitions or solves again.
    # Later failures are caught by step!'s `|| break`.
    if !SciMLBase.successful_retcode(it.buffer[it.state])
        return SolutionWrapper(it.buffer, it.state)
    end
    while !Base.isdone(it)
        CommonSolve.step!(it) || break
    end
    return SolutionWrapper(it.buffer, it.state)
end
