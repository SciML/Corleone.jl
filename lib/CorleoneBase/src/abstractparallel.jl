abstract type AbstractParallelProblem end

function get_problem(::AbstractParallelProblem) end
function prob_func(::AbstractParallelProblem, template, i) end

function CommonSolve.init(
    problem::AbstractParallelProblem, args...;
    kwargs...
)
    ensemble_problem = EnsembleProblem(
        get_problem(problem), 
        prob_func = Base.Fix1(prob_func, problem), 
        kwargs...
    )
    CommonSolve.solve(ensemble_problem, args...; kwargs...)
end