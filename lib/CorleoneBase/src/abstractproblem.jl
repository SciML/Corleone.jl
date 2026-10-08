abstract type AbstractSequentialProblem end

function get_problem(::AbstractSequentialProblem) end

function transition(::AbstractSequentialProblem, sol, i) end

function terminal(::AbstractSequentialProblem, sol, i) end

# Out-of-place solvers require the state and derivative to use the same array
# representation. AD broadcasts may return a struct-of-arrays representation.
prepare_stage_problem(problem) = problem
function prepare_stage_problem(problem::SciMLBase.AbstractODEProblem)
    SciMLBase.isinplace(problem) && return problem
    u0 = ArrayInterface.aos_to_soa(problem.u0)
    p = ArrayInterface.aos_to_soa(problem.p)
    u0 === problem.u0 && p === problem.p && return problem
    return remake(problem; u0, p)
end

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

"""
    init(problem::AbstractSequentialProblem, algorithm; kwargs...)

Remake the initial problem with any supplied `u0`, `p`, and `tspan` keywords.
Forward all remaining keywords to every stage's solve. Later stage problems
are supplied by the transition, not remade with the initial overrides.
"""
function CommonSolve.init(
        problem::AbstractSequentialProblem, algorithm;
        kwargs...
    )
    initial_kwargs = (;
        (
            key => value for (key, value) in kwargs
                if key in (:u0, :p, :tspan)
        )...,
    )
    solve_kwargs = Base.structdiff((; kwargs...), initial_kwargs)
    inner_problem = prepare_stage_problem(remake(get_problem(problem); initial_kwargs...))
    sol = solve(inner_problem, algorithm; solve_kwargs...)
    SciMLBase.successful_retcode(sol) || throw(
        ErrorException(
            "The initial call to solve failed with returncode $(sol.retcode)"
        )
    )
    buffer = make_buffer(problem, sol)
    return SequentialProblemIterator(
        inner_problem,
        Base.Fix1(transition, problem),
        Base.Fix1(terminal, problem),
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
    SciMLBase.successful_retcode(current_solution) || return false
    if length(buffer) >= state + 1
        buffer[state + 1] = current_solution
    else
        push!(buffer, current_solution)
    end
    increment!(it)
    return true
end

function Base.isdone(it::SequentialProblemIterator, state = it.state)
    return it.terminal(it.buffer[state], state)
end


function CommonSolve.solve!(it::SequentialProblemIterator)
    Base.isdone(it) && return it
    while !Base.isdone(it)
        CommonSolve.step!(it) || break
    end
    return it
end
