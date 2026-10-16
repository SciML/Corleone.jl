using LuxCore
using Random
using ConcreteStructs
using ChainRulesCore


function lookup!(cache, ::typeof(searchsortedfirst), t, timegrid)
    get!(cache, t) do 
        searchsortedfirst(timegrid, t, lt = <=)
    end
end

function lookup!(::Nothing, ::typeof(searchsortedfirst), t, timegrid)
    searchsortedfirst(timegrid, t, lt = <=)
end

function lookup!(cache, ::typeof(searchsorted), t, timegrid)
    get!(cache, t) do 
        only(searchsorted(timegrid, t))
    end
end

function lookup!(::Nothing, ::typeof(searchsorted), t, timegrid)
    only(searchsorted(timegrid, t))
end

@concrete terse struct LookUp{F} <: LuxCore.AbstractLuxLayer
    "The lookup function"
    lookup::F
    "The initial timegrid"
    timegrid
    "The (optional) cache"
    cache 
end

LuxCore.initialparameters(::Random.AbstractRNG, ::LookUp) = (; )
LuxCore.initialstates(::Random.AbstractRNG, x::LookUp) = (; 
    timegrid = x.timegrid, 
    inserted = Int64[],
)

function LuxCore.apply(model::LookUp, t::T, ::Any, st::NamedTuple) where T <: Real
    (; lookup, cache) = model 
    (; timegrid) = st 
    return lookup!(cache, lookup, t, timegrid), st 
end


@concrete terse struct ParameterCollection <: LuxCore.AbstractLuxLayer
    outdims 
    init_parameters
end

LuxCore.initialparameters(rng::Random.AbstractRNG, x::ParameterCollection) = (; ps = x.init_parameters(rng, outdims))
LuxCore.initialstates(::Random.AbstractRNG, ::ParameterCollection) = (;)

function LuxCore.apply(::ParameterCollection, index, ps, st)
    return getindex(ps.ps, index), st    
end

@concrete terse struct PiecewiseConstantParameter <: LuxCore.AbstractLuxContainerLayer{(:lookup, :parameter)}
    name 
    lookup 
    parameter
end




@non_differentiable lookup!(cache, f, t, timegrid)

lookup!(nothing, searchsortedfirst, 3.2, [1., 2., 3.2])