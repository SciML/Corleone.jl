"""
$(TYPEDEF)

A simple wrapper for a function from ModelingToolkit.
"""
struct DynamicsFunction{F,P,R,S} <: LuxCore.AbstractLuxContainerLayer{(:dynamics, :controls)}
    "The dynamics of the problem"
    dynamics::F
    "The initial parameters"
    p::P
    "Remake for the parameters"
    re::R
    "The prototype for constants etc"
    states::S

    function DynamicsFunction(prob::SciMLBase.DEProblem)
        p, re, st = __extract_params(prob.p)
        f = prob.f
        return new{typeof(f),typeof(p),typeof(re),typeof(st)}(f, p, re, st)
    end
end

# For MTK we overload this for MTK paramters
__extract_params(p::AbstractArray) = (p, identity, nothing)

LuxCore.initialstates(::Random.AbstractRNG, f::DynamicsFunction) = deepcopy(f.states)
LuxCore.initialparameters(::Random.AbstractRNG, f::DynamicsFunction) = deepcopy(f.p)

function (model::DynamicsFunction{<:Any,<:Any,<:Any,Nothing})(args::Tuple, ps, st)
    t, rest... = Base.reverse(args)
    p = model.re(ps)
    out = model.dynamics(Base.reverse(rest)..., p, t)
    return out, st
end

function (model::DynamicsFunction)(args::Tuple, ps, st)
    t, rest... = Base.reverse(args)
    p = (model.re(ps), st...)
    out = model.dynamics(Base.reverse(rest)..., p, t)
    return out, st
end

function merge_initials(x, y, replaces)
    @assert size(x) == size(y) "x and y have to be equally sized!"
    [j ∈ replaces ? y[j] : x[j] for j in eachindex(x)]
end

struct ShootingProblem{P,D,I,A,E} <: LuxCore.AbstractLuxContainerLayer{(:dynamics, :intervals)}
    "The underlying DEProblem"
    problem::P
    "The new dynamics function"
    dynamics::D
    # These are just wrappers for u0s, parameters, and tspans
    "The shooting intervals"
    intervals::I
    "The algorithm"
    algorithm::A
    "The ensemble algorithm"
    ensemble_algorithm::E
end

(shoot::ShootingProblem)(::Any, ps, st) = shoot(shoot.problem.u0, ps, st)

function (shoot::ShootingProblem)(u0::AbstractArray, ps, st)
    (; problem, dynamics, intervals) = shoot
    wrapped_dynamics = wrap_model(dynamics, ps.dynamics, st.dynamics)
end



