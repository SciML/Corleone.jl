"""
$(TYPEDEF)

Defines a function `f` over a trajectory. 

If `saveat` is not provided as a sorted `AbstractVector{<:Real}` with unique entries, 
it is assumed that the function is valid over the initial set of optimization variables, 
which include initial conditions and (gridded) parameters. Hence the `f` has the signature 
`f(p_global) -> res`.

If `saveat` is provided, the function `f` is assumed to be time dependendent and omits a signature
`f([du(t_i),] u(t_i), p(t_i), t_i) -> res`, where `du` is available in the case of DAEs. 

If no `lower_bounds` or `upper_bounds`, it is assumed that `f` is not a constraint. 

# Fields 
$(FIELDS)
"""
struct GridFunction{SAVEAT, IIP, CS, N, S} <: AbstractTimeGridLayer{false, SAVEAT}
    "The name of the constraint"
    name::N
    "The wrapped initial states"
    initial_states::S
end

SciMLBase.isinplace(::GridFunction{<:Any, IIP}) where IIP = IIP 
isa_constraint(::GridFunction{<:Any, <:Any, CS}) where CS = CS 
LuxCore.initialparameters(::Random.AbstractRNG, ::GridFunction) = NamedTuple()
LuxCore.parameterlength(::GridFunction) = 0
LuxCore.initialstates(rng::Random.AbstractRNG, layer::GridFunction) = layer.initial_states(rng)
 

function GridFunction{IIP}(
    f::Function,
    saveat::AbstractVector{<:Real} = Float64[]; 
    name = nothing,
    lower_bounds = nothing, 
    upper_bounds = nothing, 
) where IIP
    
    if !isempty(saveat) 
        SAVEAT = true
        @assert issorted(saveat) "The provided timepoints should be sorted."
        @assert unique(saveat) == saveat "The provided timepoints are not unique."
    else 
        SAVEAT = false
    end

    if !isnothing(lower_bounds) || !isnothing(upper_bounds)
        @assert typeof(lower_bounds) == typeof(upper_bounds) "The type of the bounds is inconsistent."
        @assert size(lower_bounds) == size(upper_bounds) "The size of the bounds is inconsistent."
        @assert all(lower_bounds .<= upper_bounds) "The lower bounds ≴ upper bounds."
        #if !isempty(saveat)
        #    @assert size(lower_bounds) == size(saveat) "The size of the bounds and timepoints `saveat` is inconsistent."
        #end
        CS = true
    else
        CS = false
    end

    initial_states = let saveat = saveat, lower_bounds = lower_bounds, upper_bounds = upper_bounds
        (rng) -> NamedTuple{(:f, :saveat, :lower_bounds, :upper_bounds,)}(
            (f, _maybecopy(saveat), _maybecopy(lower_bounds), _maybecopy(upper_bounds))
        )
    end

    GridFunction{SAVEAT, IIP, CS, typeof(name), typeof(initial_states)}(
        name, initial_states
    )
end

# Global, OOP 
function (::GridFunction{false, false})(p, ::Any, st::NamedTuple)
    (; f) = st 
    return f(p), st
end

# Global, IIP
function (::GridFunction{false, true})((res, p)::Tuple, ::Any, st::NamedTuple)
    (; f) = st 
    return f(res, p), st
end

# Time Dependent, OOP
function (::GridFunction{true, false})(x::Base.AbstractVecOrTuple, ::Any, st::NamedTuple)
    (; f) = st 
    res = reduce(vcat, map(eachindex(x)) do i 
        @views f(x[i]...)
    end)
    return res, st
end

function (::GridFunction{true, true})(x::Base.AbstractVecOrTuple, ::Any, st::NamedTuple)
    (; f) = st 
    foreach(eachindex(x)) do i 
        @views f(x[i]...)
    end
    return x, st
end

