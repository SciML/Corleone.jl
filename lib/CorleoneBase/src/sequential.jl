"""
    SequentialProblem(problem; transition = (sol, i) -> nothing,
                      terminal = (sol, i) -> i >= 1)

Solve a sequence of SciML problems with `CommonSolve.init`/`solve` and an algorithm.
The terminal predicate receives `(solution, current_index)`, starting at index 1.
If it is false, `transition(solution, next_index)` supplies the next SciML problem,
starting at index 2. State, parameters, and time spans are not propagated
automatically: use e.g. `remake(sol.prob; u0 = sol.u[end], tspan = ..., p = ...)`
in the transition. Returned problems must support the same algorithm and their
solutions must be compatible with the iterator's solution buffer element type.

`CommonSolve.init` solves the first stage immediately. `CommonSolve.step!` solves
one next stage and returns its success flag; it does not check termination or
previous-stage success. `CommonSolve.solve!` runs until the terminal predicate
returns true or a stage fails, retaining the failed stage. An unsuccessful first
stage never transitions during `solve!`. Solver and callback exceptions propagate.
Supply a terminal condition that eventually stops the sequence; the default
stops after one stage and the default transition supplies no next problem.

The completed
iterator stores stage solutions in `buffer` and the current index in `state`.
`CommonSolve.solve!` and `CommonSolve.solve` return a read-only `SolutionWrapper`
(an `AbstractVector`) over the solved stages, whose `retcode` is the last solved
stage's exact return code; `CommonSolve.init` returns the iterator itself.
Initial keywords that name a field of the initial problem (for an `ODEProblem`:
`u0`, `p`, `tspan`, `f`, `kwargs`, `problem_type`) remake only the first stage,
with `ArrayInterface.aos_to_soa` applied to their values; remaining keywords are
forwarded unchanged to every stage. Keep endpoint saving enabled when propagating
`sol.u[end]`. `maxiters` limits each stage's solver, not the total number of
stages.
"""
struct SequentialProblem{P, T, R} <: AbstractSequentialProblem
    "The initial problem to solve (first stage)"
    problem::P
    "Transition callable that receives `(solution, next_index)`
  and returns the next `AbstractSciMLProblem`"
    transition::T
    "Terminal predicate callable that receives `(solution, current_index)`
  and returns `Bool`. When `true`, no further stages are solved."
    terminal::R
end

get_problem(x::SequentialProblem) = x.problem
transition(x::SequentialProblem, sol, i) = x.transition(sol, i)
terminal(x::SequentialProblem, sol, i) = x.terminal(sol, i)

function SequentialProblem(
        problem::SciMLBase.AbstractSciMLProblem;
        transition = (sol, i) -> nothing,
        terminal = (sol, i) -> i >= 1,
    )
    return SequentialProblem{typeof(problem), typeof(transition), typeof(terminal)}(
        problem, transition, terminal
    )
end
