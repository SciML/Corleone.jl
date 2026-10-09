struct ParallelProblem{P, F} <: AbstractParallelProblem
    problem::P
    prob_func::F
end

get_problem(x::ParallelProblem) = x.problem
prob_func(x::ParallelProblem, template, i) = x.prob_func(x, template, i)

ParallelProblem(problem; prob_func = (x, prob, i) -> prob) =
    ParallelProblem(problem, prob_func)
