"""
$(TYPEDEF)

Defines a callable layer representing a piecewise constant control. The control is defined by a `DiffEqArray` containing values `u` at the timepoints `t`.

# Fields
$(FIELDS)
"""
struct PiecewiseConstantControl{S,C,B,SHOOT} <: LuxCore.AbstractLuxLayer
    "The symbolic index of the control"
    idx::S
    "The control timeseries"
    signal::C
    "The bounds for the control values"
    bounds::B
end

function PiecewiseConstantControl(sym, signal::DiffEqArray, bounds=(to_val(signal.u, -Inf), to_val(signal.u, Inf)), shooting=false)
    @assert length(signal.u) == length(signal.t) "The length of control values must match the length of timepoints."
    @assert issorted(signal.t) "Timepoints must be sorted in ascending order."
    @assert all(isfinite, signal.u) "Control values must be finite."
    @assert all(Base.Fix1(Base.broadcast, isfinite), bounds) "Bounds must be finite."
    PiecewiseConstantControl{typeof(sym),typeof(signal),typeof(bounds),shooting}(sym, signal, bounds)
end

PiecewiseConstantControl(sym, u, t, bounds=(to_val(u, -Inf), to_val(u, Inf)), shooting=false) = PiecewiseConstantControl(sym, DiffEqArray(u, t), bounds, shooting)

is_shooted(::PiecewiseConstantControl{<:Any,<:Any,<:Any,SHOOT}) where SHOOT = SHOOT

Base.nameof(control::PiecewiseConstantControl) = Symbol(control.idx)
LuxCore.display_name(control::PiecewiseConstantControl) = nameof(control)

function Base.getproperty(control::PiecewiseConstantControl, name::Symbol)
    name === :u ? getfield(control, :signal).u :
    name === :t ? getfield(control, :signal).t :
    getfield(control, name)
end

function get_lower_bound(layer::PiecewiseConstantControl)
    b = first(layer.bounds)
    size(b) == size(layer.u) ? b : fill(b, length(layer.t))
end
function get_upper_bound(layer::PiecewiseConstantControl)
    b = last(layer.bounds)
    size(b) == size(layer.u) ? b : fill(b, length(layer.t))
end

SciMLBase.remake(control::PiecewiseConstantControl; kwargs...) = begin
    signal = get(kwargs, :signal, DiffEqArray(get(kwargs, :u, control.u), get(kwargs, :t, control.t)))
    tspan = get(kwargs, :tspan, extrema(signal.t))
    bounds = get(kwargs, :bounds, control.bounds)
    _clamp_tspan(PiecewiseConstantControl(control.idx, signal, bounds), tspan)
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
        DiffEqArray(u[mask], t[mask]),
        deepcopy.(control.bounds),
        shoot
    )
end

_maybevec(x::AbstractVector) = x
_maybevec(x) = vec(x)
_maybevec(x::Number) = [x]

LuxCore.parameterlength(control::PiecewiseConstantControl) = length(control.signal) * product(size(first(control.u)))
LuxCore.initialparameters(::Random.AbstractRNG, control::PiecewiseConstantControl) = (; 
    u = clamp.(control.u, get_lower_bound(control), get_upper_bound(control))
)

LuxCore.initialstates(::Random.AbstractRNG, control::PiecewiseConstantControl) = (; 
    t = control.t,
    current=firstindex(control.t),
    lower=firstindex(control.t), upper=lastindex(control.t))
LuxCore.statelength(control::PiecewiseConstantControl) = 3 + length(control.t)

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
struct ProblemRemaker{P,U,BU,TP,BP,Q,HAS_U0,HAS_P} <: LuxCore.AbstractLuxLayer
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
    has_u0 = !isempty(tunable_u0)
    has_p = !isempty(tunable_p)
    ProblemRemaker{
        typeof(problem),typeof(tunable_u0),typeof(u0_bounds),typeof(tunable_p),
        typeof(p_bounds),typeof(quadrature_indices),has_u0,has_p}(
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
    m = which(SciMLBase.remake, (typeof(problem),))
    kw = Base.kwarg_decl(m)
    if !isempty(kw)
        _kwargs = (; (k => v for (k, v) in pairs(kwargs) if k in kw)...)
    else
        drop = Set((:problem, :tunable_u0, :tunable_p, :u0_bounds, :p_bounds, :quadrature_indices))
        _kwargs = (; (k => v for (k, v) in pairs(kwargs) if !(k in drop))...)
    end
    problem = remake(problem; _kwargs...)
    has_u0 = !isempty(tunable_u0)
    has_p = !isempty(tunable_p)
    ProblemRemaker{typeof(problem),typeof(tunable_u0),typeof(u0_bounds),typeof(tunable_p),typeof(p_bounds),typeof(quadratures),has_u0,has_p}(problem, tunable_u0, u0_bounds, tunable_p, p_bounds, quadratures)
end

get_problem(layer::ProblemRemaker) = layer.problem
get_tspan(layer::ProblemRemaker) = layer.problem.tspan
get_quadrature_indices(layer::ProblemRemaker) = layer.quadrature_indices
get_tunable_u0(layer::ProblemRemaker) = layer.tunable_u0
get_tunable_p(layer::ProblemRemaker) = layer.tunable_p

get_lower_bound(layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,true,true}) = (; u0=first(layer.u0_bounds), p=first(layer.p_bounds))
get_lower_bound(layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,true,false}) = (; u0=first(layer.u0_bounds))
get_lower_bound(layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,false,true}) = (; p=first(layer.p_bounds))
get_lower_bound(layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,false,false}) = NamedTuple()

get_upper_bound(layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,true,true}) = (; u0=last(layer.u0_bounds), p=last(layer.p_bounds))
get_upper_bound(layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,true,false}) = (; u0=last(layer.u0_bounds))
get_upper_bound(layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,false,true}) = (; p=last(layer.p_bounds))
get_upper_bound(layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,false,false}) = NamedTuple()

LuxCore.initialparameters(::Random.AbstractRNG, layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,true,true}) = begin
    T = promote_type(eltype(layer.problem.u0), eltype(layer.problem.p))
    (;
        u0=convert.(T, clamp.(getsym(layer.problem, layer.tunable_u0)(layer.problem), layer.u0_bounds...)),
        p=convert.(T, clamp.(getsym(layer.problem, layer.tunable_p)(layer.problem), layer.p_bounds...))
    )
end

LuxCore.initialparameters(::Random.AbstractRNG, layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,true,false}) = begin
    T = promote_type(eltype(layer.problem.u0), eltype(layer.problem.p))
    (; u0=convert.(T, clamp.(getsym(layer.problem, layer.tunable_u0)(layer.problem), layer.u0_bounds...)))
end

LuxCore.initialparameters(::Random.AbstractRNG, layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,false,true}) = begin
    T = promote_type(eltype(layer.problem.u0), eltype(layer.problem.p))
    (; p=convert.(T, clamp.(getsym(layer.problem, layer.tunable_p)(layer.problem), layer.p_bounds...)))
end

LuxCore.initialparameters(::Random.AbstractRNG, layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,false,false}) = NamedTuple()

function LuxCore.parameterlength(layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,U0,P}) where {U0,P}
    length(get_tunable_u0(layer)) + length(get_tunable_p(layer))
end

LuxCore.initialstates(::Random.AbstractRNG, ::ProblemRemaker) = NamedTuple()
LuxCore.statelength(::ProblemRemaker) = 0


_get_u0(layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,U}, problem, ps) where U =
    if U
        __remake_wrap(problem, problem.u0, layer.tunable_u0, ps.u0)
    else
        problem.u0
    end

_get_p(layer::ProblemRemaker{<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,<:Any,P}, problem, ps) where P =
    if P
        __remake_wrap(problem, problem.p, layer.tunable_p, ps.p)
    else
        problem.p
    end


function (layer::ProblemRemaker)(::Any, ps, st)
    (; problem) = layer
    remake(problem, u0=_get_u0(layer, problem, ps), p=_get_p(layer, problem, ps)), st
end
