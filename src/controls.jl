"""
$(TYPEDEF)
Implements a piecewise constant control discretization.

# Fields
$(FIELDS)

# Examples

```julia
using Corleone

# Define a control parameter with time points and default settings
control = ControlParameter(0.0:0.1:10.0)
# Define a control parameter with custom control values and bounds
control = ControlParameter(0.0:0.1:10.0; name = :u, controls = (rng, t) -> rand(rng, length(t)), bounds = t -> (zeros(length(t)), ones(length(t))))
```
"""
struct ControlParameter{T,C,B,SHOOTED} <: LuxCore.AbstractLuxLayer
    "The name of the control"
    name::Symbol
    "The timepoints at which discretized variables are introduced. If empty, we assume a single value constant over time."
    t::T
    "The initial values for the controls in form of a function (rng, t) -> values. Defaults to [`default_controls`](@ref)."
    controls::C
    "The bounds for the control values in form of a function (t) -> (lower_bounds, upper_bounds). Defaults to `nothing`, which corresponds to unbounded controls derived from the controls."
    bounds::B

    function ControlParameter(t::AbstractVector; name::Symbol=gensym(:u), controls::Function=default_controls, bounds::Union{Nothing,Function}=nothing, shooted::Bool=false)
        return new{typeof(t),typeof(controls),typeof(bounds),shooted}(name, t, controls, bounds)
    end
end

"""
$(FUNCTIONNAME)

Default constructor for the control values of a [`ControlParameter`](@ref). 
The control values are initialized as zeros of the same length and element type as the time vector `t`.

$(SIGNATURES)
"""
default_controls(rng, t::AbstractVector) = isempty(t) ? zeros(Float64, 1) : zeros(eltype(t), size(t)...)

"""
$(FUNCTIONNAME)

Checks if the control is shooted, i.e., if it has a value which will be constrained via an equality constraint in the optimization problem.
"""
is_shooted(::ControlParameter{<:Any,<:Any,<:Any,SHOOTED}) where SHOOTED = SHOOTED

get_lower_bound(layer::ControlParameter{<:Any,<:Any,Nothing}) = to_val(layer.controls(Random.default_rng(), layer.t), -Inf)
get_upper_bound(layer::ControlParameter{<:Any,<:Any,Nothing}) = to_val(layer.controls(Random.default_rng(), layer.t), Inf)

get_lower_bound(layer::ControlParameter{<:Any,<:Any,<:Function}) = first(layer.bounds(layer.t))
get_upper_bound(layer::ControlParameter{<:Any,<:Any,<:Function}) = last(layer.bounds(layer.t))

get_bounds(layer::ControlParameter) = (get_lower_bound(layer), get_upper_bound(layer))

ControlParameter(x::Base.Pair{Symbol,<:AbstractVector}) = ControlParameter(last(x), name=first(x))
ControlParameter(x::Base.Pair{Symbol,<:Base.AbstractRange}) = ControlParameter(collect(last(x)), name=first(x))

ControlParameter(x::Base.Pair{Symbol,<:NamedTuple}) = begin
    nt = last(x)
    ControlParameter(getproperty(nt, :t),
        name=first(x),
        controls=get(nt, :controls, default_controls),
        bounds=get(nt, :bounds, nothing),
        shooted=get(nt, :shooted, false),
    )
end

ControlParameter(x::ControlParameter) = x
ControlParameter(x) = throw(ArgumentError("Invalid argument for ControlParameter constructor: $x"))


_maybeextrema(t) = isempty(t) ? (0.0, 0.0) : extrema(t) 

function SciMLBase.remake(layer::ControlParameter;
    name::Symbol=layer.name,
    controls::Function=layer.controls,
    bounds::Function=layer.bounds,
    t::AbstractVector=deepcopy(layer.t),
    tspan=_maybeextrema(t),
    shooted = false,
    kwargs...
)

    mask = zeros(Bool, length(t))

    if isempty(t)
        return ControlParameter(t; name, controls, bounds, shooted)
    end
    if tspan == _maybeextrema(t)
        mask .= true
    else
        t0, tinf = tspan
        for i in eachindex(t)
            if t[i] >= t0 && t[i] < tinf
                mask[i] = true
            end
            if i != lastindex(t) && t[i] < t0 && t[i+1] > t0
                mask[i] = true
                t[i] = t0
                shooted = true
            end
        end
    end

    ControlParameter( t[mask]; name, controls, bounds, shooted)
end

LuxCore.initialparameters(rng::Random.AbstractRNG, control::ControlParameter) = begin
    lb, ub = Corleone.get_bounds(control)
    #(;
    controls = clamp.(control.controls(rng, control.t), lb, ub)
    #)
end

LuxCore.initialstates(::Random.AbstractRNG, control::ControlParameter) = (;
    t=control.t,
    current_index=firstindex(control.t),
    first_index=firstindex(control.t),
    last_index=lastindex(control.t),
    # TODO Add a fixed size hash table lookup here to avoid the linear search in find_idx for large control grids
    # Maybe build a tree structure 
)

find_idx(t::T, timepoints::AbstractVector) where T<:Number = searchsortedlast(timepoints, t)

function (::ControlParameter)(tcurrent, controls, st::NamedTuple)
    (; t, current_index, first_index, last_index) = st
    if current_index == last_index && tcurrent >= t[last_index]
        return controls[current_index], st
    elseif current_index == first_index == last_index # Constant control case 
        return controls[current_index], st
    end
    current_index = clamp(find_idx(tcurrent, t), first_index, last_index)
    return controls[current_index], merge(st, (; current_index))
end

"""
$(TYPEDEF)

A collection of control parameters, which can be used to define multiple controls in a structured way. 
The controls are stored in a named tuple, where the keys correspond to the control names and the values are the control parameters. 
The `transform` field can be used to apply a transformation to the control values before they are returned by the layer.

# Fields
$(FIELDS)

# Examples
```julia
using Corleone
# Define multiple control parameters with custom settings
controls = ControlParameters(
    :u => 0.0:0.1:10.0,
    :v => 0.0:0.2:10.0;
    transform = (cs) -> (u = cs.u, v = cs.v)
)
controls = ControlParameters(
    :u => 0.0:0.1:10.0,
    ControlParameter(:v, 0.0:0.2:10.0, controls = (rng, t) -> rand(rng, length(t)), bounds = t -> (zeros(length(t)), ones(length(t))));
    transform = (cs) -> (u = cs.u, v = cs.v)
)
```
"""
struct ControlParameters{C<:NamedTuple,T} <: LuxCore.AbstractLuxWrapperLayer{:controls}
    "The name of the container"
    name::Symbol
    "The control parameter collection"
    controls::C
    "The ouput transformation"
    transform::T
end

function ControlParameters(controls::NamedTuple; name::Symbol=gensym(:controls), transform=identity, kwargs...)
    return ControlParameters{typeof(controls),typeof(transform)}(name, controls, transform)
end

function ControlParameters(controls...; kwargs...)
    controls = map(ControlParameter, controls)
    names = map(c -> c.name, controls)
    controls = NamedTuple{names}(controls)
    return ControlParameters(controls; kwargs...)
end

function (layer::ControlParameters)(tnow, ps, st)
    (; transform) = layer
    cs, st = _eval_controls(layer.controls, tnow, ps, st)
    return transform(cs), st
end

@generated function _eval_controls(controls::NamedTuple{fields}, t, ps, st) where {fields}
    returns = [gensym() for _ in fields]
    rt_states = [gensym() for _ in fields]
    expr = Expr[]
    for (i, sym) in enumerate(fields)
        push!(expr, :(($(returns[i]), $(rt_states[i])) = controls.$(sym)(t, ps.$(sym), st.$(sym))))
    end
    push!(expr,
        :(st = NamedTuple{$fields}((($(Tuple(rt_states)...),))))
    )
    push!(expr, :(result = NamedTuple{$fields}(($(returns...),))))
    push!(expr, :(return result, st))
    ex = Expr(:block, expr...)
    return ex
end
