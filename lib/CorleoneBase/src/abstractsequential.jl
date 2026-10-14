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
[`SequentialProblem`](@ref). Concrete problems call their stored hook with the
sequence itself as the first argument, so the hook can read sequence-wide data
such as `seq.parameters` and `seq.states`.
"""
function transition(x::AbstractSequentialProblem, sol, i) 
    get_problem(x)
end

"""
    terminal(problem::AbstractSequentialProblem, sol, i)

Return `true` when stage `i`'s solution `sol` ends the sequence (`i` starts at
1). Implemented by concrete sequential problems; see [`SequentialProblem`](@ref).
Concrete problems call their stored hook with the sequence itself as the first
argument, so the hook can read sequence-wide data such as `seq.parameters` and
`seq.states`.
"""
function terminal(::AbstractSequentialProblem, sol, i) 
    i >= 1 || !SciMLBase.successful_retcode(sol)
end

function output(::AbstractSequentialProblem, sol, i)
    sol
end

function get_size(::AbstractSequentialProblem) end

has_size(::AbstractSequentialProblem) = false

"""
    make_buffer(problem::AbstractSequentialProblem, sol::T) -> Vector{T}
return T[sol]h the first stage's solution `sol`.
The element type is `T`, the concrete type of `sol`, so every later stage must
produce that exact type.
"""
function make_buffer(x::AbstractSequentialProblem, sol::T) where {T}
    if has_size(x)
        buffer = Vector{T}(undef, get_size(x))
        buffer[1] = sol 
        return buffer
    else 
        return T[sol]
    end
    #if !isnothing(Base.get_extension(CorleoneBase, :CorleoneBaseZygoteExtension)) 
    #    return make_zygote_buffer(x, buffer)
    #else
    #    return buffer
    #end
end



@generated function _prepare_problem(x::P) where {P}
    f_names = fieldnames(P)
    isempty(f_names) && return :(x)
    conv_exprs = [
        :($(Symbol(:conv_, f)) = ArrayInterface.aos_to_soa(getfield(x, $(QuoteNode(f)))))
            for f in f_names
    ]
    comparisons = [
        :(getfield(x, $(QuoteNode(f))) === $(Symbol(:conv_, f))) for f in f_names
    ]
    unchanged = reduce((acc, c) -> :($acc && $c), comparisons)
    kw_exprs = [Expr(:kw, f, Symbol(:conv_, f)) for f in f_names]
    return quote
        $(conv_exprs...)
        $unchanged && return x
        return remake($(Expr(:parameters, kw_exprs...)), x)
    end
end

@inline function _is_inplace(x::SciMLBase.AbstractSciMLProblem)
    return SciMLBase.isinplace(x) === true
end

@inline function prepare_stage_problem(x::SciMLBase.AbstractSciMLProblem)
    _is_inplace(x) && return x
    return _prepare_problem(x)
end

@inline prepare_stage_problem(problem) = problem

@generated function _split_sequence_kwargs(
        ::S, nt_kwargs::NamedTuple{K}
    ) where {S <: AbstractSequentialProblem, K}
    # These execute during compilation:
    valid_keys = fieldnames(S)
    seq_keys = Tuple(k for k in K if k in valid_keys)
    rest_keys = Tuple(k for k in K if !(k in valid_keys))

    # This generates the exact type-stable slicing code for the specific kwargs passed
    return quote
        seq_kwargs = NamedTuple{$seq_keys}(nt_kwargs)
        rest_kwargs = NamedTuple{$rest_keys}(nt_kwargs)
        return seq_kwargs, rest_kwargs
    end
end


@inline function split_sequence_kwargs(prob::AbstractSequentialProblem, kwargs)
    return _split_sequence_kwargs(prob, NamedTuple(kwargs))
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
mutable struct SequentialProblemIterator{P, N, O, T, A, B, S, M}
    "The SciMLProblem to solve in a sequence"
    const problem::P
    "The transition function"
    const transition::N
    "The output function"
    const output::O
    "The terminal condition"
    const terminal::T
    "The algorithm to solve"
    const algorithm::A
    "The buffer"
    const buffer::B
    "The current iteration"
    current::S
    "The maximum estimated iteration"
    max_steps::M
end


@inline function CommonSolve.init(
        problem::T, algorithm::SciMLBase.AbstractSciMLAlgorithm; 
        kwargs...
    ) where {T <: AbstractSequentialProblem}

    seq_kwargs, kwargs = split_sequence_kwargs(problem, kwargs)
    sequence = isempty(seq_kwargs) ? problem : remake(problem; seq_kwargs...)
    inner_problem = prepare_stage_problem(
        remake(
            transition(sequence, nothing, 1); 
            kwargs...
            )
        )
    sol = solve(inner_problem, algorithm; dense = false, save_everystep = false, save_start = true ,save_end = true)
    bff = output(sequence, sol, 1)
    buffer = make_buffer(sequence, bff)
    return SequentialProblemIterator(
        inner_problem,
        (sol, i) -> transition(sequence, sol, i),
        (sol, i) -> output(sequence, sol, i),
        (sol, i) -> terminal(sequence, sol, i),
        algorithm,
        buffer,
        1,
        has_size(sequence) ? get_size(sequence) : nothing
    )
end

function increment!(it::SequentialProblemIterator)
    it.current += 1
    return it
end

function CommonSolve.step!(it::SequentialProblemIterator)
    (; algorithm, transition, output, buffer, current) = it
    next = current+1
    current_problem = prepare_stage_problem(transition(buffer[current], next))
    if length(buffer) >= next
        buffer[next] = output(solve(current_problem, algorithm; save_start = false, save_end = true), next)
    else
        push!(buffer, output(solve(current_problem, algorithm; save_start = false, save_end = true), next))
    end
    increment!(it)
end

function Base.isdone(it::SequentialProblemIterator)
    return it.terminal(it.buffer[it.current], it.current)
end

finalize_buffer!(it::SequentialProblemIterator) = finalize_buffer!(it.buffer, it)

function finalize_buffer!(buffer::AbstractVector, x::SequentialProblemIterator)
    (; current) = x 
    resize!(buffer, current)
end

function CommonSolve.solve!(it::SequentialProblemIterator{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, Nothing})
    while true 
        Base.isdone(it) && break
        CommonSolve.step!(it) 
    end
    return finalize_buffer!(it)
end


function CommonSolve.solve!(it::SequentialProblemIterator{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Int})
    (; max_steps, buffer, transition, output, algorithm) = it
    for current in 1:max_steps
        Base.isdone(it) && break
        next = current+1
        current_problem = prepare_stage_problem(transition(buffer[current], next))
        buffer[next] = output(solve(current_problem, algorithm; save_start = false, save_end = true), next)
        increment!(it)
    end
    return finalize_buffer!(it)
end