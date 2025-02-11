# Check the bounds provided from the initialization
function check_parameterbounds(p, bounds, id::String)
    isnothing(bounds) && return  
    @assert typeof(p) === typeof(bounds) "The type of the $id bounds does not math the type of the parameters."
    @assert size(p) == size(bounds) "The size of the $id bounds does not math the type of the parameters."
    if id == "lower"
        @assert all(p .>= bounds) "The initial values are not consistent with the $id bound."
    else
        @assert all(p .<= bounds) "The initial values are not consistent with the $id bound."
    end
    return  
end

"""
$(TYPEDEF)

A struct which models a parameter value. If no `tstops` are given, the call will always return `p`.
If the optional keyword `tstops` is set with a vector of time stops, it will behave like a piecewise constant 
function over `t` in the first dimension of the provided `p <: AbstractArray`.
"""
struct Parameter{ISDISCRETE, N, P, S} <: AbstractTimeGridLayer{ISDISCRETE, false} 
    "The name of the layer"
    name::N
    "The wrapped initial parameters"
    initial_parameters::P
    "The wrapped initial states"
    initial_states::S
end 

LuxCore.initialparameters(rng::Random.AbstractRNG, layer::Parameter) = layer.initial_parameters(rng)
LuxCore.initialstates(rng::Random.AbstractRNG, layer::Parameter) = layer.initial_states(rng)

function Parameter(
        p::AbstractArray{T};
        name = nothing,
        tstops = nothing,
        lower_bounds = nothing, 
        upper_bounds = nothing,
        kwargs...
    ) where T

    @assert !isempty(size(p)) "No initial parameters have been passed to the constructor. Please consider using a function in the dynamics instead."
    
    check_parameterbounds(p, lower_bounds, "lower")
    check_parameterbounds(p, upper_bounds, "upper")
    
    tstops = maybe_unique_sort(tstops)

    if isa(tstops, AbstractVector)
        @assert length(tstops) == size(p, 1) "The initial parameters are not consistent with the provided tstops."
    else 
        @assert isnothing(tstops) "The provided tstops are of type $typeof(tstops), but need to be either `nothing` or an `AbstractArray`"
    end 

    initial_parameters = let p = p 
        (rng) -> NamedTuple{(:p,)}((_maybecopy(p),))
    end

    initial_states = let tstops = tstops, lower_bounds = lower_bounds, upper_bounds = upper_bounds
        (rng) -> NamedTuple{(:tstops, :lower_bounds, :upper_bounds)}(
            (_maybecopy(tstops), _maybecopy(lower_bounds), _maybecopy(upper_bounds))
        )
    end

    Parameter{!isnothing(tstops), typeof(name), typeof(initial_parameters), typeof(initial_states)}(
        name, initial_parameters, initial_states
    )
end

# Helper function for finding the right parameter value
function find_index(t, val)
    idx = searchsortedlast(t, val)
    # We always assume that we have left / right continuity
    min(max(firstindex(t), idx), lastindex(t)) 
end

function find_index(p, t, val)
    @assert size(t, 1) == size(p, 1) "The dimensionality of the provided `tstops` and `parameters` are not consistent."
    id = find_index(t, val)
    selectdim(p, 1, id)
end

function (params::Parameter{true})(t, ps, st)
    (; tstops) = st 
    (; p) = ps
    return find_index(p, tstops, t), st
end

function (params::Parameter{false})(t, ps, st)
    (; p) = ps
    return selectdim(p, 1, :), st
end
