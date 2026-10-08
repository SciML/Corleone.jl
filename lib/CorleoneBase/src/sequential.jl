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
