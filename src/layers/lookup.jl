"""
$(SIGNATURES)

Look up the interval or entry for `key`, optionally caching the result.

- `cache` is a cache for storing already looked up keys. In case of `nothing` the caching is skipped.
- `collection` either an `AbstractVector` or `AbstractDict` used for the lookup
- `key` the key.

If used with an `AbstractVector` and `Number`, lookup defaults to piecewise constant interpolation using
`searchsortedfirst(collection, key; lt = <=)`: a switching time belongs to the
following interval. Dictionary lookup requires an exact key and returns
`typemax` of its value type when absent. A `nothing` collection returns index 1.
"""
function lookup!(cache, collection, key::T) where {T}
    return get!(cache, key) do
        lookup!(nothing, collection, key)
    end
end

function lookup!(::Nothing, collection::AbstractVector{T}, key::V) where {T, V}
    return searchsortedfirst(collection, key, lt = <=)
end

function lookup!(::Nothing, collection::AbstractDict{T, V}, key::K) where {T, V, K}
    return get(collection, key, typemax(V))
end

function lookup!(::Nothing, collection::AbstractDict{Any, Any}, key::K) where {K}
    return get(collection, K(key), typemax(Int))
end

lookup!(::Nothing, ::Nothing, ::T) where {T} = 1

@non_differentiable lookup!(cache, collection, key)

"""
$(TYPEDEF)

A time-based lookup layer using [`lookup!`](@ref). Its collection, optional cache,
and inserted-index buffer are stored in Lux state during setup.

# Fields
$(FIELDS)

"""
@concrete terse struct LookUp <: LuxCore.AbstractLuxLayer
    "The initial switching-time vector, exact-time dictionary, or `nothing`."
    collection
    "The optional dictionary used to cache time lookups, or `nothing`."
    cache
end

LuxCore.initialparameters(::Random.AbstractRNG, ::LookUp) = (;)
LuxCore.initialstates(::Random.AbstractRNG, x::LookUp) = (;
    collection = x.collection,
    inserted = Int64[],
    cache = x.cache,
)

function LuxCore.apply(::LookUp, t::T, ::Any, st::NamedTuple) where {T <: Real}
    (; collection, cache, inserted) = st
    idx = lookup!(cache, collection, t)
    return (idx, idx in inserted), st
end

Base.haskey(::LookUp, k, ps, st) = _haskey(st.collection, k)
_haskey(x::Base.AbstractDict, k) = haskey(x, k)
_haskey(::Any, ::Any) = false

get_timestops(::LookUp, ps, st, ::Type{T} = Float64) where {T} = get_timestops(st.collection, T)
get_timestops(x::AbstractDict, ::Type{T}) where {T} = T.(collect(keys(x)))
get_timestops(x::AbstractVector, ::Type{T}) where {T} = T.(x)
get_timestops(::Nothing, ::Type{T}) where {T} = T[]
