struct ProblemLayer

"""
$(TYPEDEF)

A struct representing a single shooting problem. 

# Fields 
$(FIELDS)
"""
struct ShootingProblem{P,U} <: LuxCore.AbstractLuxLayer
    "The (tunable) parameter of the problem"
    p::P
    "The initial conditions of the problem"
    u0::U
    "The problem kwargs which should change, e.g. tspan or saveat."
    kwargs::NamedTuple
    "The tunable initial conditions"
    tunable::Vector{UInt64}
end

function ShootingProblem(problem::SciMLBase.DEProblem; tunable=eachindex(problem.u0), kwargs...)
    ShootingProblem(problem.p, problem.u0, NamedTuple(kwargs), UInt64.(tunable))
end

LuxCore.initialparameters(rng::Random.AbstractRNG, interval::ShootingProblem) = (; u0=copy(interval.u0), p=deepcopy(interval.p))
LuxCore.initialstates(rng::Random.AbstractRNG, interval::ShootingProblem) = (;
    tunable=copy(interval.tunable),
    kwargs=deepcopy(interval.kwargs)
)

function merge_initials(x, y, replaces)
    @assert size(x) == size(y) "x and y have to be equally sized!"
    [j ∈ replaces ? y[j] : x[j] for j in eachindex(x)]
end

function (prob::ShootingProblem)(problem::SciMLBase.DEProblem, ps, st)
    (; u0, p) = ps
    (; tunable, kwargs) = st
    u0_ = merge_initials(problem.u0, u0, tunable)
    problem = remake(problem; u0=u0_, p=p, kwargs...), st
end

struct MultipleShootingProblem{P<:NamedTuple} <: LuxCore.AbstractLuxContainerLayer{(:problems,)}
    "The single shooting problems"
    problems::P
end

function (prob::MultipleShootingProblem)(problem::SciMLBase.DEProblem, ps, st)
    (; problems) = prob
    remaker = let names = keys(problems), ps = ps.problems, st = st.problems, problems = problems
        function (prob, i, repeat)
            current = names[i]
            p_current = getproperty(ps, current)
            st_current = getproperty(st, current)
            prob_current = getproperty(problems, current)
            first(prob_current(prob, p_current, st_current))
        end
    end
    return EnsembleProblem(problem, prob_func=remaker), st
end

struct SolverLayer{A,E,S} <: LuxCore.AbstractLuxWrapperLayer{:shooting}
    "The algorithm to solve the problem with"
    algorithm::A
    "The ensemble algorithm to solve a multiple shooting problem"
    ensemble_algorithm::E
    "The single or multiple shooting problem"
    shooting::S 
end

function (solver::SolverLayer{<:Any, <:Any, <:ShootingProblem})(problem::DEProblem, ps, st)
    (; algorithm, shooting) = solver 
    new_prob, _ = shooting(problem, ps, st)
    solve(new_prob, algorithm), st
end

function (solver::SolverLayer{<:Any, <:Any, <:MultipleShootingProblem})(problem::DEProblem, ps, st)
    (; algorithm, ensemble_algorithm, shooting) = solver 
    new_prob, _ = shooting(problem, ps, st)
    solve(new_prob, algorithm, ensemble_algorithm; trajectories = length(st.problems)), st
end