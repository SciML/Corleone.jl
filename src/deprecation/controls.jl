"""
$(TYPEDEF)
Implements a piecewise constant control discretization.

# Fields
$(FIELDS)
"""
struct ControlParameter{T, C, B}
    "The name of the control"
    name::Symbol
    "The timepoints at which discretized variables are introduced"
    t::T
    "The initial values for the controls. Either a vector or a function (rng,t,bounds) -> u"
    controls::C
    "The bounds as a tuple"
    bounds::B
end

default_u(rng, t, bounds) = zeros(eltype(t), size(t))
default_bounds(t::AbstractVector{T}) where {T <: Real} = (fill(typemin(T), size(t)), fill(typemax(T), size(t)))

"""
$(SIGNATURES)

Constructs a `ControlParameter` with piecewise constant discretizations introduced at
timepoints `t`. Optionally

```julia-repl
julia> ControlParameter(0:1.0:4.0, name=:c)
ControlParameter{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, typeof(Corleone.default_u), typeof(Corleone.default_bounds)}(:c, 0.0:1.0:10.0, Corleone.default_u, Corleone.default_bounds)
```

```julia-repl
julia> ControlParameter(0.0:1.0:4.0, name=:c, controls = zeros(5))
ControlParameter{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, Vector{Float64}, typeof(Corleone.default_bounds)}(:c, 0.0:1.0:4.0, [0.0, 0.0, 0.0, 0.0, 0.0], Corleone.default_bounds)
```


```julia-repl
julia> ControlParameter(0:1.0:9.0, name=:c1, controls=zeros(5), bounds=(0.0,1.0))
ControlParameter{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, Vector{Float64}, Tuple{Float64, Float64}}(:c1, 0.0:1.0:9.0, [0.0, 0.0, 0.0, 0.0, 0.0], (0.0, 1.0))
```
The latter is functionally equivalent to the following example, specifying all bounds individually:
```julia-repl
julia> ControlParameter(0:1.0:9.0, name=:c1, controls=zeros(5), bounds=(zeros(5),ones(5)))
ControlParameter{StepRangeLen{Float64, Base.TwicePrecision{Float64}, Base.TwicePrecision{Float64}, Int64}, Vector{Float64}, Tuple{Vector{Float64}, Vector{Float64}}}(:c1, 0.0:1.0:9.0, [0.0, 0.0, 0.0, 0.0, 0.0], ([0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0, 1.0]))
```
"""
function ControlParameter(t::AbstractVector; name::Symbol = gensym(:w), controls = default_u, bounds = default_bounds)
    return ControlParameter{typeof(t), typeof(controls), typeof(bounds)}(name, t, controls, bounds)
end

"""
$(SIGNATURES)

Return the control time points inside `tspan`. The returned points retain the order
of the control parameter grid and exclude the right endpoint of a finite interval.
"""
get_timegrid(parameters::ControlParameter, tspan = (-Inf, Inf)) = begin
    (; t) = parameters
    idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
    t[idx]
end

"""
$(SIGNATURES)

Computes the number of discretized controls restricted to given `tspan`.
"""
function control_length(parameters::ControlParameter; tspan = nothing, kwargs...)
    (; t) = parameters
    idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
    return size(idx, 1)
end

"""
$(SIGNATURES)

Returns discretized controls of ControlParameter `params` restricted to given `tspan`.
"""
function get_controls(::Random.AbstractRNG, parameters::ControlParameter{<:Any, <:AbstractArray}; raw = false, tspan = nothing, kwargs...)
    (; t, controls) = parameters
    raw && return controls
    idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
    return controls[idx]
end

function get_controls(rng::Random.AbstractRNG, parameters::ControlParameter{<:Any, <:Function}; raw = false, tspan = nothing, kwargs...)
    (; t) = parameters
    bounds = get_bounds(parameters; tspan, kwargs...)
    idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
    return parameters.controls(rng, t[idx], bounds)
end

"""
$(SIGNATURES)

Returns bounds of discretized controls restricted to given `tspan`.
"""
function get_bounds(parameters::ControlParameter{<:Any, <:Any, <:Tuple}; tspan = nothing, kwargs...)
    (; t) = parameters
    idx = isnothing(tspan) ? eachindex(t) : findall(tspan[1] .<= t .< tspan[2])
    nc = size(idx, 1)
    _bounds = parameters.bounds
    if length(_bounds[1]) == length(_bounds[2]) == 1
        return (repeat([_bounds[1]], nc), repeat([_bounds[2]], nc))
    elseif length(_bounds[1]) == length(_bounds[2]) == length(t)
        return (_bounds[1][idx], _bounds[2][idx])
    end
    throw("Incompatible control bound definition. Got $(length(_bounds[1])) elements, expected $(length(t)).")
end

get_bounds(parameters::ControlParameter{<:Any, <:Any, <:Function}; tspan = (-Inf, Inf), kwargs...) = parameters.bounds(get_timegrid(parameters, tspan; kwargs...))

function check_consistency(rng::Random.AbstractRNG, parameters::ControlParameter)
    grid = get_timegrid(parameters)
    u = get_controls(rng, parameters; raw = true)
    lb, ub = get_bounds(parameters)
    @assert issorted(grid) "Time grid is not sorted."
    @assert get_timegrid(parameters) == unique(grid) "Time grid is not unique."
    @assert all(lb .<= ub) "Bounds are inconsistent"
    @assert size(lb) == size(ub) == size(u) == size(grid) "Sizes are inconsistent"
    return @assert all(lb .<= u .<= ub) "Initial values are inconsistent"
end
