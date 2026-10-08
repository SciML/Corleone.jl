"""
$(TYPEDEF)

`SequentialProblem` wraps an initial SciML problem and additional transitions and
terminals to solve it as a sequence of stages.

Fields:
- `initialproblem::P` - The initial problem to solve (first stage).
- `transition::T` - Transition callable that receives `(solution, completed_index)`
  and returns the next `AbstractSciMLProblem`.
- `terminal::R` - Terminal predicate callable that receives `(solution, completed_index)`
  and returns `Bool`. When `true`, no further stages are solved.

Constructor: `SequentialProblem(problem::AbstractSciMLProblem; transition, terminal)`
"""
abstract type AbstractSequentialProblem end 

function get_problem(::AbstractSequentialProblem) end 

function transition(::AbstractSequentialProblem, sol, i) end

function terminal(::AbstractSequentialProblem, sol, i) end

Base.IteratorSize(::AbstractSequentialProblem) = Base.SizeUnknown()

wrap_buffer(::Val, buffer) = buffer

function make_buffer(sol::T, ::Any, ::AbstractSequentialProblem) where T
    T[sol]
end

function make_buffer(sol::T, ::Base.HasLength, x::AbstractSequentialProblem) where T
    buffer = Vector{T}(undef, length(x))
    buffer = wrap_buffer(Val(:Zygote), buffer)
    buffer[1] = sol 
    buffer
end

mutable struct SequentialProblemIterator{P, N, T, A, B, S}
    const problem::P
    const transition::N 
    const terminal::T
    const algorithm::A
    const buffer::B
    state::S
end


function CommonSolve.init(problem::AbstractSequentialProblem, algorithm; 
    kwargs...)
    inner_problem = remake(get_problem(problem); kwargs...)
    sol = solve(inner_problem, algorithm)
    buffer = make_buffer(sol, Base.IteratorSize(problem), problem) 
    SequentialProblemIterator(
        inner_problem,
        Base.Fix1(transition, problem),
        Base.Fix1(terminal, problem), 
        algorithm, 
        buffer, 
        1
        )
end

function CommonSolve.step!(it::SequentialProblemIterator)
    (; algorithm, transition, buffer, state) = it
    current_problem = transition(buffer[state], state+1)
    if length(buffer) >= state+1
        buffer[state+1] = solve(current_problem, algorithm)
    else 
        push!(buffer, solve(current_problem, algorithm))
    end 
    it.state += 1
    return it
end

function Base.isdone(it::SequentialProblemIterator, state = it.state)
    it.terminal(it.buffer[state], state)
end

function CommonSolve.solve!(it::SequentialProblemIterator)
    Base.isdone(it) && return it
    while !Base.isdone(it) 
        CommonSolve.step!(it)
    end
    return it
end
