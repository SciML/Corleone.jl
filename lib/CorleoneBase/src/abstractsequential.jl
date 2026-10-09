abstract type AbstractSequentialProblem <: SciMLBase.AbstractSciMLProblem end

function get_problem(::AbstractSequentialProblem) end

function transition(::AbstractSequentialProblem, sol, i) end

function terminal(::AbstractSequentialProblem, sol, i) end


function make_buffer(::AbstractSequentialProblem, sol::T) where {T}
    return T[sol]
end


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
@generated function _split_problem_kwargs(::P, nt_kwargs::NamedTuple{K}) where {P <: SciMLBase.AbstractSciMLProblem, K}
    # These execute during compilation:
    valid_keys = fieldnames(P)
    prob_keys  = Tuple(k for k in K if k in valid_keys)
    solve_keys = Tuple(k for k in K if !(k in valid_keys))
    
    # This generates the exact type-stable slicing code for the specific kwargs passed
    return quote
        prob_kwargs  = map(ArrayInterface.aos_to_soa, NamedTuple{$prob_keys}(nt_kwargs))
        solve_kwargs = NamedTuple{$solve_keys}(nt_kwargs)
        return prob_kwargs, solve_kwargs
    end
end

@generated function _prepare_problem(x::P) where P
    f_names = fieldnames(P)
    kw_exprs = [
        Expr(:kw, f, :(ArrayInterface.aos_to_soa(getfield(x, $(QuoteNode(f))))))
        for f in f_names
    ]
    return Expr(:call, :remake, Expr(:parameters, kw_exprs...), :x)
end

@inline prepare_stage_problem(x::SciMLBase.AbstractSciMLProblem) = _prepare_problem(x) 

# 2. Your frontend function simply converts the kwargs to a NamedTuple and forwards it
@inline function split_problem_kwargs(prob::SciMLBase.AbstractSciMLProblem, kwargs)
    return _split_problem_kwargs(prob, NamedTuple(kwargs))
end

"""
    init(problem::AbstractSequentialProblem, algorithm; kwargs...)

Remake the initial problem with any supplied `u0`, `p`, and `tspan` keywords.
Forward all remaining keywords to every stage's solve. Later stage problems
are supplied by the transition, not remade with the initial overrides.
"""
@inline function CommonSolve.init(
        problem::T, algorithm::SciMLBase.AbstractSciMLAlgorithm;
        kwargs...
    ) where T <: SciMLBase.AbstractSciMLProblem

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
