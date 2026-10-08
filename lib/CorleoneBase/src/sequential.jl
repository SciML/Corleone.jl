"""
    SequentialProblem(problem; transition = (sol, i) -> nothing,
                      terminal = (sol, i) -> i >= 1)

Solve a sequence of SciML problems with `CommonSolve.init`/`solve` and an algorithm.
The terminal predicate receives the current solution and stage index. If it is
false, `transition(solution, next_index)` supplies the next problem. The completed
iterator stores stage solutions in `buffer` and the current index in `state`.
Initial `u0`, `p`, and `tspan` solve keywords remake only the first problem;
remaining keywords are forwarded to every stage.
"""
struct SequentialProblem{P, T, R} <: AbstractSequentialProblem
    "The initial problem to solve (first stage)"
    problem::P
    "Transition callable that receives `(solution, current_index)`
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
    SequentialProblem{typeof(problem), typeof(transition), typeof(terminal)}(
        problem, transition, terminal
    )
end
