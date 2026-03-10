"""
$(TYPEDEF)

A struct containing the problem definition for a dynamic optimization problem. 

# Fields
$(FIELDS)

# Examples
```julia
using Corleone
using OrdinaryDiffEq 
prob = ODEProblem((u, p, t) -> -p[1] .* u, [1.0, 0.0], (0.0, 10.0), [0.5])
layer = InitialCondition(prob, name = :linear_problem, tunable_ic = [1, ])
``` 
"""
struct InitialCondition{P, B} <: LuxCore.AbstractLuxLayer
    "The name of the layer"
    name::Symbol
    "The <:DEProblem defining the dynamics"
    problem::P
    "The indices of the initial condition that are tunable parameters in the optimization problem"
    tunable_ic::Vector{Int}
    "The bounds of the initial conditions. Expects either nothing for unbounded parameters or a function of the form (t0) -> (lower_bounds, upper_bounds)."
    bounds::B
    "Additional quadrature indices if present."
    quadrature_indices::Vector{Int}
end

get_lower_bound(layer::InitialCondition{<:Any, Nothing}) = to_val(layer.problem.u0[layer.tunable_ic], -Inf)
get_upper_bound(layer::InitialCondition{<:Any, Nothing}) = to_val(layer.problem.u0[layer.tunable_ic], Inf)

get_lower_bound(layer::InitialCondition{<:Any, <:Function}) = first(layer.bounds(layer.problem.tspan[1]))
get_upper_bound(layer::InitialCondition{<:Any, <:Function}) = last(layer.bounds(layer.problem.tspan[1]))

get_bounds(layer::InitialCondition) = (get_lower_bound(layer), get_upper_bound(layer))

get_tspan(layer::InitialCondition) = layer.problem.tspan

get_timegrid(layer::InitialCondition) = begin
    tspan = collect(layer.problem.tspan)
    saveats = get(layer.problem.kwargs, :saveat, eltype(tspan)[])
    vcat(tspan, saveats)
end

function InitialCondition(prob::SciMLBase.DEProblem; name::Symbol = gensym(:problem), tunable_ic = Int[], bounds::Union{Nothing, Function} = nothing, quadrature_indices = Int[])
    @assert isempty(setdiff(tunable_ic, eachindex(prob.u0))) "Tunable initial condition indices must be within the bounds of the initial condition vector."
    @assert isempty(setdiff(quadrature_indices, eachindex(prob.u0))) "Quadrature indices must be within the bounds of the initial condition vector."
    @assert isempty(intersect(tunable_ic, quadrature_indices)) "Tunable initial condition indices and quadrature indices must be disjoint."

    return InitialCondition{typeof(prob), typeof(bounds)}(name, prob, tunable_ic, bounds, quadrature_indices)
end

LuxCore.initialparameters(::Random.AbstractRNG, layer::InitialCondition) = begin
    (; problem) = layer
    (; u0) = problem
    (; tunable_ic) = layer
    deepcopy(u0[tunable_ic])
end

LuxCore.parameterlength(layer::InitialCondition) = length(layer.tunable_ic)

LuxCore.initialstates(::Random.AbstractRNG, layer::InitialCondition) = begin
    (; problem, tunable_ic, quadrature_indices) = layer
    (; u0) = problem
    keeps = [i ∉ tunable_ic for i in eachindex(u0)]
    replaces = zeros(Bool, length(u0), length(tunable_ic))
    for (i, idx) in enumerate(tunable_ic)
        replaces[idx, i] = true
    end
    return (; u0 = deepcopy(u0), keeps, replaces, quadrature_indices)
end

function (layer::InitialCondition)(::Any, ps, st::NamedTuple)
    (; problem) = layer
    (; u0, keeps, replaces) = st
    u0_new = keeps .* u0 .+ replaces * ps
    return SciMLBase.remake(problem, u0 = u0_new), st
end

function SciMLBase.remake(
        layer::InitialCondition;
        name::Symbol = layer.name,
        problem::SciMLBase.DEProblem = layer.problem,
        tunable_ic::Vector{Int} = layer.tunable_ic,
        bounds = layer.bounds,
        quadrature_indices::Vector{Int} = layer.quadrature_indices,
        kwargs...,
    )
    m = which(SciMLBase.remake, (typeof(problem),))
    kw = Base.kwarg_decl(m)
    _kwargs = NamedTuple()
    if !isempty(kw)
        _kwargs = (; (k => v for (k, v) in pairs(kwargs) if k in kw)...)
    end
    problem = remake(problem; _kwargs...)
    return InitialCondition(problem; name, tunable_ic, bounds, quadrature_indices)
end

#= 
"""
$(SIGNATURES)

Initializes all shooting nodes with random values.

# Arguments
- `rng::Random.AbstractRNG` a random number generator
- `shooting::MultipleShootingLayer` a shooting layer

# Keyworded Arguments
- `ps` the default parameters of the `shooting` layer.
"""
function random_initialization(
        rng::Random.AbstractRNG,
        shooting::MultipleShootingLayer;
        ps = default_initialization(rng, shooting),
        kwargs...,
    )
    (; layer) = shooting
    (; tunable_ic) = layer
    u0 = last(ps).u0
    isempty(u0) && return ps
    lb, ub = get_bounds(shooting)
    vals = map(enumerate(zip(ps, lb, ub))) do (i, (plocal, lbi, ubi))
        unew = if i == 1
            _random_value(rng, lbi.u0[tunable_ic], ubi.u0[tunable_ic])
        else
            _random_value(rng, lbi.u0, ubi.u0)
        end
        merge(plocal, (; u0 = unew))
    end
    return NamedTuple{keys(ps)}(vals)
end

"""
$(SIGNATURES)

Linearly interpolates u0 and u_inf for t with tspan[1] < t < tspan[2].

# Arguments
- `rng::Random.AbstractRNG` a random number generator
- `shooting::MultipleShootingLayer` a shooting layer

# Keyworded Arguments
- `ps` the default parameters of the `shooting` layer.
- `u_infinity::AbstractArray` the value at the last timepoint for all states.
"""
function linear_initialization(
        rng::Random.AbstractRNG,
        shooting::MultipleShootingLayer;
        ps = default_initialization(rng, shooting),
        u_infinity = last(ps).u0,
        kwargs...,
    )
    (; shooting_intervals, layer) = shooting
    (; problem, tunable_ic) = layer
    isempty(u_infinity) && return ps
    u0 = first(ps).u0
    if isempty(u0)
        u0 = problem.u0
    end
    t0, tinf = problem.tspan
    foreach(enumerate(ps)) do (i, sps)
        t = shooting_intervals[i][1]
        u0new = (u_infinity .- u0) .* (t - t0) ./ tinf .+ u0
        sps.u0 .= i > 1 ? u0new : u0new[tunable_ic]
    end
    return ps
end

__setu0!(u0::AbstractArray, ::Any...) = u0

function __setu0!(u0::AbstractArray, unew::AbstractArray, tunable_ics)
    return unew[tunable_ics]
end

function __setu0!(u0::AbstractArray, unew::Dict, tunable_ics)
    return map(tunable_ics) do i
        get(unew, i, u0[i])
    end
end

function __setu0!(u0::AbstractArray, unew::Base.AbstractVecOrTuple{<:Pair}, tunable_ics)
    return map(tunable_ics) do i
        id = findfirst(==(i), first.(unew))
        isnothing(id) && u0[i]
        last.(unew)[id]
    end
end

"""
$(SIGNATURES)


Initializes all shooting nodes with user-provided values. Initial values are given as an iterable collection of
`AbstractArray`s, `Dict`s, or a vector of `Pair`s.  The variable indices are interpreted depending on the passed value

- `AbstractArray` just sets the initial condition with the provided value. This assumes equal dimensionality.
- `Dict` assumes the keys represent indices of the initial condition.
- A `Vector` or `Tuple` of `Pair`s assumes the first value represents an index of the initial condition.

Other options simply skip the corresponding interval.

# Arguments
- `rng::Random.AbstractRNG` a random number generator
- `shooting::MultipleShootingLayer` a shooting layer

# Keyworded Arguments
- `ps` the default parameters of the `shooting` layer.
- `u0s` the collection of initial conditions.

# Example
```julia
# Assumes a shooting layer with 3 intervals
# Skips the first interval
# Sets the first index of the second interval to 3.0
# Sets the second index of the third interval to 5.0 and the 5th index to -1.0
custom_initialization(rng, shooting, u0s = [nothing, Dict(1 => 3.0), (2 => 5.0, 5 => -1.0)])
```
"""
function custom_initialization(
        rng::Random.AbstractRNG,
        shooting::MultipleShootingLayer;
        ps = default_initialization(rng, shooting),
        u0s = [],
        kwargs...,
    )
    (; layer) = shooting
    (; tunable_ic) = layer
    isempty(u0s) && return ps
    foreach(enumerate(ps)) do (i, sps)
        if lastindex(u0s) >= i
            sps.u0 .= __setu0!(sps.u0, u0s[i], i > 1 ? eachindex(sps.u0) : tunable_ic)
        end
    end
    return ps
end

"""
$(SIGNATURES)

Initializes the problem using a forward solve of the problem. This results in a continuous
trajectory.

# Arguments
- `rng::Random.AbstractRNG` a random number generator
- `shooting::MultipleShootingLayer` a shooting layer

# Keyworded Arguments
- `ps` the default parameters of the `shooting` layer.
"""
function forward_initialization(
        rng::Random.AbstractRNG,
        shooting::MultipleShootingLayer;
        ps = default_initialization(rng, shooting),
        fixed_indices = Int[],
        kwargs...,
    )
    (; layer) = shooting
    st = LuxCore.initialstates(rng, shooting)
    u0s = [first(ps).u0]
    for (i, (sps, sst)) in enumerate(zip(ps, st))
        u0 = i ∈ fixed_indices ? sps.u0 : last(u0s)
        sps.u0 .= u0[eachindex(sps.u0)]
        pred, _ = layer(nothing, sps, sst)
        u0_ = pred.u[end]
        push!(u0s, u0_)
    end
    return ps
end

"""
$(SIGNATURES)


Initializes all shooting nodes with user-provided value. The variable indices are interpreted depending on the passed value

- `AbstractArray` just sets the initial condition with the provided value. This assumes equal dimensionality.
- `Dict` assumes the keys represent indices of the initial condition.
- A `Vector` or `Tuple` of `Pair`s assumes the first value represents an index of the initial condition.

Other options simply skip the corresponding interval.

# Arguments
- `rng::Random.AbstractRNG` a random number generator
- `shooting::MultipleShootingLayer` a shooting layer

# Keyworded Arguments
- `ps` the default parameters of the `shooting` layer.
- `u0` the initial condition.

# Example
```julia
# Assumes a shooting layer with 3 intervals
# Sets the first index of all intervals to 3.0
constant_initialization(rng, shooting, u0 = Dict(1 => 3.0))
constant_initialization(rng, shooting, u0 = (1 => 3.0,))
# Set the initial condition to the given vector
constant_initialization(rng, shooting, u0 = [1., 2., 4.])
```
"""
function constant_initialization(
        rng::Random.AbstractRNG,
        shooting::MultipleShootingLayer;
        ps = default_initialization(rng, shooting),
        u0 = nothing,
        kwargs...,
    )
    (; layer) = shooting
    (; tunable_ic) = layer
    isempty(u0) && return ps
    foreach(enumerate(ps)) do (i, sps)
        sps.u0 .= __setu0!(sps.u0, u0, i > 1 ? eachindex(sps.u0) : tunable_ic)
    end
    return ps
end

"""
$(SIGNATURES)

Initializes the shooting nodes in a hybrid method by applying the provided methods and indices `f`
sequentially. Here we assume the structure `interval => method` for the initialization.

# Arguments
- `rng::Random.AbstractRNG` a random number generator
- `shooting::MultipleShootingLayer` a shooting layer
- `f::Pair`s of the interval index and applied method

# Keyworded Arguments
- `ps` the default parameters of the `shooting` layer.

All other keyworded arguments are passed on to the functions below.
"""
function hybrid_initialization(
        rng::Random.AbstractRNG,
        shooting::MultipleShootingLayer,
        f::Pair...;
        ps = default_initialization(rng, shooting),
        kwargs...,
    )
    (intervals, method), rest = Base.first(f), Base.tail(f)
    pnew = method(rng, shooting; ps = deepcopy(ps), kwargs...)
    names = ntuple(i -> keys(ps)[intervals[i]], size(intervals, 1))
    pnew = merge(ps, NamedTuple{names}(pnew))
    return hybrid_initialization(rng, shooting, rest...; ps = pnew, kwargs...)
end

function hybrid_initialization(
        rng::Random.AbstractRNG,
        shooting::MultipleShootingLayer;
        ps = default_initialization(rng, shooting),
        kwargs...,
    )
    return ps
end
 =#
