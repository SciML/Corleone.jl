__isa_wrapped(::StatefulWrapper) = true
__isa_wrapped(::Any) = false

function remake_timepoints(ps::NamedTuple{T}, st::NamedTuple{T}, tspan::Tuple) where T
    isempty(T) && return ps, st
    psst = map(T) do name
        remake_timepoints(getproperty(ps, name), getproperty(st, name), tspan)
    end
    NamedTuple(zip(T, first.(psst))), NamedTuple(zip(T, last.(psst)))
end

remake_timepoints(x,y,z) = x,y

function remake_timepoints(ps::NamedTuple, st::NamedTuple, (t0, tinf)::Tuple)
    :timepoints ∈ keys(st) || return ps, st
    idmin = searchsortedlast(st.timepoints, t0)
    idmax = searchsortedlast(st.timepoints, prevfloat(tinf))
    idx = idmin:idmax
    ps = merge(ps, (; local_controls=ps.local_controls[idx]))
    st = merge(st, (; timepoints=st.timepoints[idx], max_index=length(idx)))
    ps, st
end

function controlled_remake(prob::P where P<:SciMLBase.DEProblem; kwargs...)
    prob = remake(prob; kwargs...)
    __isa_wrapped(prob.f.f) || return prob
    # Get the original model, which is a ControlledFunction 
    ps = NamedTuple(prob.p)
    st = NamedTuple(prob.f.f.state)
    ps, st = remake_timepoints(ps, st, prob.tspan)
    pnew = ComponentArray(ps)
    newf = deepcopy(prob.f.f)
    newf.state = st
    remake(prob, f=newf, p=pnew)
end


"""
$(TYPEDEF)

A struct representing a single shooting problem. 

# Fields 
$(FIELDS)
"""
struct ShootingProblem{P,U} <: LuxCore.AbstractLuxLayer
    "The initial problem"
    problem::P
    "The problem kwargs which should change, e.g. tspan or saveat."
    kwargs::NamedTuple
    "The tunable initial conditions"
    tunable::Vector{UInt64}
end

function ShootingProblem(problem::SciMLBase.DEProblem; tunable=eachindex(problem.u0), kwargs...)
    prob = controlled_remake(problem; kwargs...)
    ShootingProblem{typeof(prob), typeof(NamedTuple(kwargs))}(prob, NamedTuple(kwargs), UInt64.(tunable))
end


LuxCore.initialparameters(rng::Random.AbstractRNG, interval::ShootingProblem) = (; u0=copy(interval.problem.u0), p=deepcopy(interval.problem.p))
LuxCore.initialstates(rng::Random.AbstractRNG, interval::ShootingProblem) = (;
    tunable=copy(interval.tunable),
)

function merge_initials(x, y, replaces)
    @assert size(x) == size(y) "x and y have to be equally sized!"
    [j ∈ replaces ? y[j] : x[j] for j in eachindex(x)]
end

(prob::ShootingProblem)(::Any, ps, st) = prob(prob.problem, ps, st)

function (prob::ShootingProblem)(problem::SciMLBase.DEProblem, ps, st)
    (; u0, p) = ps
    (; tunable) = st
    u0_ = merge_initials(problem.u0, u0, tunable)
    problem = remake(problem; u0=u0_, p=p, prob.kwargs...), st
end

struct MultipleShootingProblem{P<:NamedTuple} <: LuxCore.AbstractLuxContainerLayer{(:problems,)}
    "The single shooting problems"
    problems::P
end


function (prob::MultipleShootingProblem)(problem::Any, ps, st)
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

function (solver::SolverLayer{<:Any,<:Any,<:ShootingProblem})(problem::Any, ps, st)
    (; algorithm, shooting) = solver
    new_prob, _ = shooting(problem, ps, st)
    solve(new_prob, algorithm), st
end

function (solver::SolverLayer{<:Any,<:Any,<:MultipleShootingProblem})(problem::Any, ps, st)
    (; algorithm, ensemble_algorithm, shooting) = solver
    new_prob, _ = shooting(problem, ps, st)
    solve(new_prob, algorithm, ensemble_algorithm; trajectories=length(st.problems)), st
end