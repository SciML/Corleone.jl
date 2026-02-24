"""
$(TYPEDEF)

Defines a callable layer representing a piecewise constant control. The control is defined by the values `u` at the timepoints `t`.

# Fields
$(FIELDS)
"""
struct PiecewiseConstantControl{S,U,T,B,SHOOT} <: LuxCore.AbstractLuxLayer
    "The symbolic index of the control"
    idx::S
    "The initial control values"
    u::U
    "The timepoints at which the control is defined"
    t::T
    "The bounds for the control values"
    bounds::B
end

function PiecewiseConstantControl(sym, u, t, bounds=(to_val(u, -Inf), to_val(u, Inf)), shooting=false)
    PiecewiseConstantControl{typeof(sym),typeof(u),typeof(t),typeof(bounds),shooting}(sym, u, t, bounds)
end

is_shooted(::PiecewiseConstantControl{<:Any,<:Any,<:Any,<:Any,SHOOT}) where SHOOT = SHOOT

Base.nameof(control::PiecewiseConstantControl) = Symbol(control.idx)
LuxCore.display_name(control::PiecewiseConstantControl) = nameof(control)

function get_lower_bound(layer::PiecewiseConstantControl)
    b = first(layer.bounds)
    size(b) == size(layer.u) ? b : fill(b, length(layer.t))
end
function get_upper_bound(layer::PiecewiseConstantControl)
    b = last(layer.bounds)
    size(b) == size(layer.u) ? b : fill(b, length(layer.t))
end

SciMLBase.remake(control::PiecewiseConstantControl; kwargs...) = begin
    u = get(kwargs, :u, control.u)
    t = get(kwargs, :t, control.t)
    tspan = get(kwargs, :tspan, extrema(t))
    bounds = get(kwargs, :bounds, control.bounds)
    _clamp_tspan(PiecewiseConstantControl(control.idx, u, t, bounds), tspan)
end

function _clamp_tspan(control::PiecewiseConstantControl, (t0, tinf)::Tuple{<:Real,<:Real})
    t = deepcopy(control.t)
    u = deepcopy(control.u)
    mask = zeros(Bool, length(t))
    shoot = false

    for i in eachindex(t)
        if t[i] >= t0 && t[i] < tinf
            mask[i] = true
        end
        if i != lastindex(t) && t[i] < t0 && t[i+1] > t0
            mask[i] = true
            t[i] = t0
            shoot = true
        end
    end

    PiecewiseConstantControl(
        control.idx,
        u[mask],
        t[mask],
        deepcopy.(control.bounds),
        shoot
    )
end

_maybevec(x::AbstractVector) = x
_maybevec(x) = vec(x)
_maybevec(x::Number) = [x]

LuxCore.parameterlength(control::PiecewiseConstantControl) = length(_maybevec(control.u[1])) * length(control.t)
LuxCore.initialparameters(rng::Random.AbstractRNG, control::PiecewiseConstantControl) = (; u=clamp.(control.u, get_bounds(control)...))

LuxCore.initialstates(rng::Random.AbstractRNG, control::PiecewiseConstantControl) = (; t=control.t, current=firstindex(control.t),
    lower=firstindex(control.t), upper=lastindex(control.t))
LuxCore.statelength(control::PiecewiseConstantControl) = length(control.t) + 3

function find_idx(tcurrent, t)
    idx = searchsortedlast(t, tcurrent)
    return idx
end

function (control::PiecewiseConstantControl)(tcurrent, ps, st)
    (; t, current, lower, upper) = st
    (; u) = ps
    if t[current] <= tcurrent < t[min(current + 1, upper)]
        return u[current], st
    end
    idx = clamp(find_idx(tcurrent, t), lower, upper)
    return u[idx], merge(st, (; current=idx))
end

##

"""
$(TYPEDEF)

A helper structure to configure the initial conditions and parameters of a DEProblem.

# Fields
$(FIELDS)
"""
struct ProblemRemaker{P,U,BU,TP,BP,Q} <: LuxCore.AbstractLuxLayer
    "The underlying DE problem"
    problem::P
    "The tunable initial conditions"
    tunable_u0::U
    "The bounds for the initial conditions"
    u0_bounds::BU
    "The tunable parameters"
    tunable_p::TP
    "The bounds for the parameters"
    p_bounds::BP
    "The quadrature indices"
    quadrature_indices::Q
end

LuxCore.display_name(layer::ProblemRemaker) = hasproperty(layer.problem.f.sys, :name) ? nameof(layer.problem.f.sys) : :ProblemRemaker

function ProblemRemaker(problem;
    tunable_u0::Base.AbstractVecOrTuple=variable_symbols(problem),
    tunable_p::Base.AbstractVecOrTuple=parameter_symbols(problem),
    quadrature_indices::Base.AbstractVecOrTuple=(),
    u0_bounds::Union{Nothing,Tuple}=nothing,
    p_bounds::Union{Nothing,Tuple}=nothing,
    kwargs...
)
    # TODO Check here for Nums.  
    if isnothing(u0_bounds)
        u0b = getsym(problem, tunable_u0)(problem)
        u0_bounds = (u0b, u0b)
    end
    if isnothing(p_bounds)
        pb = getsym(problem, tunable_p)(problem)
        p_bounds = (pb, pb)
    end
    ProblemRemaker{
        typeof(problem),typeof(tunable_u0),typeof(u0_bounds),typeof(tunable_p),
        typeof(p_bounds),typeof(quadrature_indices)}(
            problem, tunable_u0, u0_bounds, tunable_p, p_bounds, quadrature_indices
            )
end

function SciMLBase.remake(remaker::ProblemRemaker; kwargs...)
    problem = get(kwargs, :problem, get_problem(remaker))
    tunable_u0 = get(kwargs, :tunable_u0, remaker.tunable_u0)
    tunable_p = get(kwargs, :tunable_p, remaker.tunable_p)
    u0_bounds = get(kwargs, :u0_bounds, remaker.u0_bounds)
    p_bounds = get(kwargs, :p_bounds, remaker.p_bounds)
    quadratures = get(kwargs, :quadrature_indices, remaker.quadrature_indices)
    problem = remake(problem; kwargs...)
    ProblemRemaker{typeof(problem),typeof(tunable_u0),typeof(u0_bounds),typeof(tunable_p),typeof(p_bounds),typeof(quadratures)}(problem, tunable_u0, u0_bounds, tunable_p, p_bounds, quadratures)
end

get_problem(layer::ProblemRemaker) = layer.problem
get_tspan(layer::ProblemRemaker) = layer.problem.tspan


get_lower_bound(layer::ProblemRemaker) = (; u0=first(layer.u0_bounds), p=first(layer.p_bounds))
get_upper_bound(layer::ProblemRemaker) = (; u0=last(layer.u0_bounds), p=last(layer.p_bounds))

LuxCore.initialparameters(rng::Random.AbstractRNG, layer::ProblemRemaker) = (;
    u0=clamp.(getsym(layer.problem, layer.tunable_u0)(layer.problem), layer.u0_bounds...),
    p=clamp.(getsym(layer.problem, layer.tunable_p)(layer.problem), layer.p_bounds...)
)

LuxCore.parameterlength(layer::ProblemRemaker) = begin
    u0 = getsym(layer.problem, layer.tunable_u0)(layer.problem)
    p = getsym(layer.problem, layer.tunable_p)(layer.problem)
    isempty(u0) ? 0 : length(u0) + (isempty(p) ? 0 : length(p))
end

LuxCore.initialstates(rng::Random.AbstractRNG, layer::ProblemRemaker) = (;)
LuxCore.statelength(::ProblemRemaker) = 0

function (layer::ProblemRemaker)(::Any, ps, st)
    (; u0, p) = ps
    (; problem) = layer
    u0_ = __remake_wrap(problem, problem.u0, layer.tunable_u0, u0)
    p_ = __remake_wrap(problem, problem.p, layer.tunable_p, p)
    remake(remake(problem, u0=u0_), p=p_), st
end
