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
struct ControlParameter{T, C, B, SHOOTED, S} <: LuxCore.AbstractLuxLayer
    "The name of the control"
    name::S
    "The timepoints at which discretized variables are introduced. If empty, we assume a single value constant over time."
    t::T
    "The initial values for the controls in form of a function (rng, t) -> values. Defaults to [`default_controls`](@ref)."
    controls::C
    "The bounds for the control values in form of a function (t) -> (lower_bounds, upper_bounds). Defaults to `nothing`, which corresponds to unbounded controls derived from the controls."
    bounds::B

    function ControlParameter(
            t::AbstractVector;
            name::N = gensym(:u),
            controls::Function = default_controls, bounds::Function = default_bounds, shooted::Bool = false,
            kwargs...
        ) where {N}
        return new{typeof(t), typeof(controls), typeof(bounds), shooted, N}(name, t, controls, bounds)
    end
end

"""
$(SIGNATURES)

Return display name used by Lux for this control parameter.
"""
LuxCore.display_name(c::ControlParameter) = Symbol(c)

"""
$(SIGNATURES)

Return symbolic control name.
"""
Base.Symbol(c::ControlParameter) = Symbol(c.name)

"""
$(FUNCTIONNAME)

Default constructor for the control values of a [`ControlParameter`](@ref). 
The control values are initialized as zeros of the same length and element type as the time vector `t`.

$(SIGNATURES)
"""
default_controls(rng, t::AbstractVector) = isempty(t) ? zeros(Float64, 1) : zeros(eltype(t), size(t)...)

"""
$(FUNCTIONNAME)

A placeholder for unbounded parameters.
"""
function default_bounds(t::AbstractVector)
    return @error "Called `default_bounds`. This should never happen!"
end

"""
$(FUNCTIONNAME)

Checks if the control is shooted, i.e., if it has a value which will be constrained via an equality constraint in the optimization problem.
"""
is_shooted(::ControlParameter{<:Any, <:Any, <:Any, SHOOTED}) where {SHOOTED} = SHOOTED

"""
$(SIGNATURES)

Return lower bounds for a [`ControlParameter`](@ref).
"""
get_lower_bound(layer::ControlParameter) = first(layer.bounds(layer.t))
get_lower_bound(layer::ControlParameter{<:Any, <:Any, <:typeof(default_bounds)}) = to_val(
    layer.controls(Random.default_rng(), layer.t), -Inf
)

"""
$(SIGNATURES)

Return upper bounds for a [`ControlParameter`](@ref).
"""
get_upper_bound(layer::ControlParameter) = last(layer.bounds(layer.t))
get_upper_bound(layer::ControlParameter{<:Any, <:Any, <:typeof(default_bounds)}) = to_val(
    layer.controls(Random.default_rng(), layer.t), Inf
)

"""
$(SIGNATURES)

Return lower and upper bounds of a [`ControlParameter`](@ref).
"""
get_bounds(layer::ControlParameter) = (get_lower_bound(layer), get_upper_bound(layer))


"""
$(SIGNATURES)

Construct a [`ControlParameter`](@ref) from a `name => timepoints` pair.
"""
ControlParameter(x::Base.Pair{Symbol, <:AbstractVector}) = ControlParameter(last(x), name = first(x))

"""
$(SIGNATURES)

Construct a [`ControlParameter`](@ref) from a `name => range` pair.
"""
ControlParameter(x::Base.Pair{Symbol, <:Base.AbstractRange}) = ControlParameter(collect(last(x)), name = first(x))

"""
$(SIGNATURES)

Construct a [`ControlParameter`](@ref) from a `name => (t, controls, bounds, shooted)` named tuple.
"""
ControlParameter(x::Base.Pair{Symbol, <:NamedTuple}) = begin
    nt = last(x)
    ControlParameter(
        getproperty(nt, :t),
        name = first(x),
        controls = get(nt, :controls, default_controls),
        bounds = get(nt, :bounds, default_bounds),
        shooted = get(nt, :shooted, false),
    )
end

"""
$(SIGNATURES)

Identity constructor for already-instantiated [`ControlParameter`](@ref).
"""
ControlParameter(x::ControlParameter) = x

"""
$(SIGNATURES)

Constructor for a `ControlParameter` with an empty time grid of element type `T`.

Creates a [`ControlParameter`](@ref) whose internal time vector `t` is initialized
as an empty `Vector{T}`. All additional keyword arguments are forwarded to the
main `ControlParameter` constructor that accepts a time vector and keyword options.
"""
function ControlParameter(::Type{T} = Float64; kwargs...) where {T <: Number}
    return ControlParameter(T[]; kwargs...)
end

$(SIGNATURES)

Throw an informative error for unsupported control constructor input.
"""
ControlParameter(x) = throw(ArgumentError("Invalid argument for ControlParameter constructor: $x"))

"""
$(SIGNATURES)

Return the time grid on which `layer` is discretized.
"""
get_timegrid(layer::ControlParameter) = layer.t

"""
$(SIGNATURES)

Return extrema of `t`, or `(0.0, 0.0)` for an empty vector.
"""
_maybeextrema(t) = isempty(t) ? (0.0, 0.0) : extrema(t)

"""
$(SIGNATURES)

Create a modified [`ControlParameter`](@ref), optionally restricting its support to `tspan`.
"""
function SciMLBase.remake(
        layer::ControlParameter;
        name = layer.name,
        controls::Function = layer.controls,
        bounds::Function = layer.bounds,
        t::AbstractVector = deepcopy(layer.t),
        tspan::Tuple{T, T} = _maybeextrema(t),
        shooted::Bool = false,
        kwargs...
    ) where {T <: Real}

    mask = zeros(Bool, length(t))

    if isempty(t)
        return ControlParameter(T[]; name, controls, bounds, shooted)
    end

    if tspan == _maybeextrema(t)
        mask .= true
    else
        t0, tinf = tspan
        for i in eachindex(t)
            if t[i] >= t0 && t[i] < tinf
                mask[i] = true
            end
            if i != lastindex(t) && t[i] < t0 && t[i + 1] > t0
                mask[i] = true
                t[i] = t0
                shooted = true
            end
        end
    end

    return ControlParameter(t[mask]; name, controls, bounds, shooted)
end

"""
$(SIGNATURES)

Initialize controls and clamp them to their bounds.
"""
LuxCore.initialparameters(rng::Random.AbstractRNG, control::ControlParameter) = begin
    lb, ub = Corleone.get_bounds(control)
    controls = map(zip(control.controls(rng, control.t), lb, ub)) do (c, l, u)
        clamp.(c, l, u)
    end
end

"""
$(SIGNATURES)

Initialize runtime state for evaluating `control`.
"""
LuxCore.initialstates(::Random.AbstractRNG, control::ControlParameter) = (;
    t = control.t,
    current_index = firstindex(control.t),
    first_index = firstindex(control.t),
    last_index = lastindex(control.t),
    # TODO Add a fixed size hash table lookup here to avoid the linear search in find_idx for large control grids
    # Maybe build a tree structure
)

"""
$(SIGNATURES)

Find the active control segment index at time `t`.
"""
find_idx(t::T, timepoints::AbstractVector) where {T <: Number} = searchsortedlast(timepoints, t)


function (::ControlParameter)(tcurrent::Number, controls, st::NamedTuple)
    (; t, current_index, first_index, last_index) = st
    isempty(t) && return only(controls), st
    if current_index == last_index && tcurrent >= t[last_index]
        return controls[current_index], st
    elseif current_index == first_index == last_index # Constant control case
        return controls[current_index], st
    end
    current_index = clamp(find_idx(tcurrent, t), first_index, last_index)
    return controls[current_index], merge(st, (; current_index))
end

"""
$(SIGNATURES)

Evaluate `layer` over all query times in `t`.
"""
function LuxCore.apply(layer::ControlParameter, t::AbstractVector, controls, st)
    ll = LuxCore.StatefulLuxLayer{true}(layer, controls, st)
    return map(Base.Fix2(ll, controls), t), ll.st
end

"""
$(TYPEDEF)

A struct which simply wraps a control parameter to allow for non-tunable tunables.
"""
struct FixedControlParameter{C <: ControlParameter} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    "The original control parameter"
    layer::C
end

function Base.getproperty(a::FixedControlParameter, v::Symbol)
    if v == :layer
        return getfield(a, :layer)
    else
        return getfield(a.layer, v)
    end
end

fix(c::ControlParameter) = FixedControlParameter{typeof(c)}(c)
FixedControlParameter(args...; kwargs...) = fix(ControlParameter(args...; kwargs...))
ControlParameter(c::FixedControlParameter) = c

LuxCore.initialparameters(::Random.AbstractRNG, ::FixedControlParameter) = (;)
LuxCore.initialstates(rng::Random.AbstractRNG, layer::FixedControlParameter) = (;
    parameters = LuxCore.initialparameters(rng, layer.layer),
    states = LuxCore.initialstates(rng, layer.layer),
)

get_lower_bound(layer::FixedControlParameter) = (;)
get_upper_bound(layer::FixedControlParameter) = (;)

function (layer::FixedControlParameter)(t, ps, st)
    return _apply_control(layer, t, ps, st)
end

function _apply_control(layer::FixedControlParameter, t, ps, st)
    out, st_ = layer.layer(t, st.parameters, st.states)
    return out, merge(st, (; states = st_))
end

SciMLBase.remake(layer::FixedControlParameter; kwargs...) = fix(SciMLBase.remake(layer.layer; kwargs...))

get_timegrid(layer::FixedControlParameter) = get_timegrid(layer.layer)

ChainRulesCore.@non_differentiable _apply_control(layer::FixedControlParameter, t, ps, st)

is_shooted(::FixedControlParameter) = false

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
struct ControlParameters{C <: NamedTuple, T} <: LuxCore.AbstractLuxWrapperLayer{:controls}
    "The name of the container"
    name::Symbol
    "The control parameter collection"
    controls::C
    "The output transformation"
    transform::T
end

"""
$(SIGNATURES)

Return the merged time grid of all controls in `layer`.
"""
get_timegrid(layer::ControlParameters) = begin
    timegrids = map(Corleone.get_timegrid, values(layer.controls))
    reduce(vcat, filter(!isempty, timegrids))
end

"""
$(SIGNATURES)

Construct [`ControlParameters`](@ref) from a named tuple of controls.
"""
function ControlParameters(controls::NamedTuple; name::Symbol = gensym(:controls), transform = identity, kwargs...)
    return ControlParameters{typeof(controls), typeof(transform)}(name, controls, transform)
end

"""
$(SIGNATURES)

Construct [`ControlParameters`](@ref) from varargs control specifications.
"""
function ControlParameters(controls...; kwargs...)
    controls = map(ControlParameter, controls)
    names = map(c -> Symbol(c), controls)
    controls = NamedTuple{names}(controls)
    return ControlParameters(controls; kwargs...)
end

"""
$(SIGNATURES)

Evaluate controls for one integration interval `(t0, tinf)`.
"""
function (layer::ControlParameters)((t0, tinf)::Tuple{T, T}, ps, st) where {T <: Number}
    (; transform) = layer
    cs, st = _apply(layer, t0, ps, st)
    return (; p = transform(cs), tspan = (t0, tinf)), st
end

"""
$(SIGNATURES)

Evaluate controls over a tuple of interval bins.
"""
function (layer::ControlParameters)(timestops::Tuple{Vararg{Tuple}}, ps, st)
    return reduce_controls(layer, ps, st, timestops), st
end

"""
$(SIGNATURES)

Apply `reducer` elementwise to a fixed-size tuple of bins.
"""
@generated function reduce_control_bin(layer, ps, st, bins::Tuple)
    N = fieldcount(bins)
    exprs = Expr[]
    rets = [gensym() for i in Base.OneTo(N)]
    for i in Base.OneTo(N)
        push!(
            exprs,
            :(($(rets[i]), st) = layer(bins[$i], ps, st))
        )
    end
    push!(exprs, Expr(:tuple, rets...))
    return Expr(:block, exprs...)
end

"""
$(SIGNATURES)

Apply `reducer` recursively to a heterogeneously-typed tuple of bins.
"""
function reduce_controls(layer, ps, st, bins::Tuple)
    current = reduce_control_bin(layer, ps, st, Base.first(bins))
    return (current, reduce_controls(layer, ps, st, Base.tail(bins))...)
end

reduce_controls(layer, ps, st, bins::Tuple{T}) where {T} = (reduce_control_bin(layer, ps, st, only(bins)),)

"""
$(SIGNATURES)

Internal helper to evaluate all controls at `tnow`.
"""
function _apply(layer::ControlParameters, tnow, ps, st)
    return _eval_controls(layer.controls, tnow, ps, st)
end

"""
$(SIGNATURES)

Generated evaluator for all controls in a named tuple.
"""
@generated function _eval_controls(controls::NamedTuple{fields}, t::T, ps, st) where {T, fields}
    returns = [gensym() for _ in fields]
    rt_states = [gensym() for _ in fields]
    expr = Expr[]
    for (i, sym) in enumerate(fields)
        push!(expr, :(($(returns[i]), $(rt_states[i])) = controls.$(sym)(t, ps.$(sym), st.$(sym))))
    end
    push!(
        expr,
        :(st = NamedTuple{$fields}((($(Tuple(rt_states)...),))))
    )
    if T <: AbstractVector
        push!(expr, :(result = map(Base.Fix1(ControlSignal, collect(t)), ($(returns...),))))
    else
        push!(expr, :(result = NamedTuple{$fields}(($(returns...),))))
    end
    push!(expr, :(return result, st))
    ex = Expr(:block, expr...)
    return ex
end

get_shooting_variables(layer::ControlParameters) = [c.name for c in values(layer.controls) if is_shooted(c)]

function SciMLBase.remake(layer::ControlParameters; kwargs...)
    name = get(kwargs, :name, layer.name)
    controls = get(
        kwargs, :controls, map(layer.controls) do control
            remake(control; kwargs...)
        end
    )
    transform = get(kwargs, :transform, layer.transform)
    return ControlParameters{typeof(controls), typeof(transform)}(name, controls, transform)
end
