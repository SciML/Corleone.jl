@concrete terse struct ParameterCollection <: LuxCore.AbstractLuxLayer
    outdims
    init_parameters
end

LuxCore.initialparameters(rng::Random.AbstractRNG, x::ParameterCollection) = (; ps = x.init_parameters(rng, x.outdims))
LuxCore.initialstates(::Random.AbstractRNG, ::ParameterCollection) = (;)

function LuxCore.apply(::ParameterCollection, index, ps, st)
    return getindex(ps.ps, index), st
end

@concrete terse struct PiecewiseConstantParameter <: LuxCore.AbstractLuxContainerLayer{(:lookup, :parameter)}
    name
    lookup
    parameter
end

function PiecewiseConstantParameter(timegrid::AbstractVector{<:Number}, default; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(searchsortedfirst, timegrid, cache)
    parameters = ParameterCollection(length(timegrid)+1, (rng, outdims) -> fill(default, outdims))
    PiecewiseConstantParameter(name, lookup, parameters)
end

function PiecewiseConstantParameter(timegrid::AbstractVector{<:Number}, default::Base.Callable; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(searchsortedfirst, timegrid, cache)
    parameters = ParameterCollection(length(timegrid) + 1, default)
    PiecewiseConstantParameter(name, lookup, parameters)
end

function (pc::PiecewiseConstantParameter)(t, ps, st)
    (; lookup, parameter) = pc 
    idx, lookup_st = LuxCore.apply(lookup, t, ps.lookup, st.lookup)
    param, parameter_st = LuxCore.apply(parameter, idx, ps.parameter, st.parameter)
    param, (; lookup = lookup_st, parameter = parameter_st)
end

@concrete terse struct InitialCondition <: LuxCore.AbstractLuxContainerLayer{(:lookup, :parameter)}
    name
    lookup
    parameter
end

function InitialCondition(timegrid::AbstractVector{<:Number}, default; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(searchsorted, timegrid, cache)
    parameters = ParameterCollection(length(timegrid), (rng, outdims) -> fill(default, outdims))
    InitialCondition(name, lookup, parameters)
end

function InitialCondition(timegrid::AbstractVector{<:Number}, default::Base.Callable; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(searchsorted, timegrid, cache)
    parameters = ParameterCollection(length(timegrid), default)
    InitialCondition(name, lookup, parameters)
end

function (pc::InitialCondition)(t, ps, st)
    (; lookup, parameter) = pc 
    idx, lookup_st = LuxCore.apply(lookup, t, ps.lookup, st.lookup)    
    param, parameter_st = LuxCore.apply(parameter, idx, ps.parameter, st.parameter)
    param, (; lookup = lookup_st, parameter = parameter_st)
end

function (pc::InitialCondition)((default, t)::Tuple, ps, st)
    (; lookup, parameter) = pc 
    idx, lookup_st = LuxCore.apply(lookup, t, ps.lookup, st.lookup) 
    idx >= typemax(typeof(idx)) && return default, (; lookup = lookup_st, parameter = st.parameter)   
    param, parameter_st = LuxCore.apply(parameter, idx, ps.parameter, st.parameter)
    param, (; lookup = lookup_st, parameter = parameter_st)
end

ic1 = InitialCondition([0., 1., 4.], [1.])

ps, st = LuxCore.setup(Random.default_rng(), ic1)
ic1((2.0, 4.1), ps, st)



using SciMLBase
using SciMLStructures

@concrete terse struct Parameters <: LuxCore.AbstractLuxWrapperLayer{(:parameters,)}
    remake
    parameters

function Parameters(prob, definitions...)
    idxs  = sortperm(collect(definitions), by = first)
    vals, rmk, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    parameters = ntuple(i->PiecewiseConstantParameter(last(definitions[idxs[i]]), vals[first(definitions[idxs[i]])]), length(definitions))
    return new{typeof(rmk), typeof(parameters)}(rmk, parameters)
end
end

LuxCore.initialparameters(rng::Random.AbstractRNG, ps::Parameters) = map(Base.Fix1(LuxCore.initialparameters, rng), ps.parameters)
LuxCore.initialstates(rng::Random.AbstractRNG, ps::Parameters) = map(Base.Fix1(LuxCore.initialstates, rng), ps.parameters)

function (p::Parameters)(tstart, ps, st)
    (; parameters, remake) = p
    results = map(eachindex(ps)) do i 
        parameters[i](tstart, ps[i], st[i])
    end
    remake(reduce(vcat, first.(results))), last.(results)
end



@concrete terse struct InitialConditions <: LuxCore.AbstractLuxWrapperLayer{(:parameters,)}
    remake
    initials

function InitialConditions(prob, definitions::Pair{Int, Any}...)
    idxs  = sortperm(collect(definitions), by = first)
    vals, rmk, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.u0)
    parameters = ntuple(i->InitialCondition(last(definitions[idxs[i]]), vals[first(definitions[idxs[i]])]), length(definitions))
    return new{typeof(rmk), typeof(parameters)}(rmk, parameters)
end
end

LuxCore.initialparameters(rng::Random.AbstractRNG, ic::InitialConditions) = map(Base.Fix1(LuxCore.initialparameters, rng), ic.initials)
LuxCore.initialstates(rng::Random.AbstractRNG, ic::InitialConditions) = map(Base.Fix1(LuxCore.initialstates, rng), ic.initials)

function (p::InitialConditions)(tstart, ps, st)
    (; initials, remake) = p
    results = map(eachindex(ps)) do i 
        initials[i](tstart, ps[i], st[i])
    end
    remake(reduce(vcat, first.(results))), last.(results)
end

function (p::InitialConditions)((default, tstart), ps, st)
    (; initials, remake) = p
    results = map(eachindex(ps)) do i 
        initials[i]((default[i], tstart), ps[i], st[i])
    end
    remake(reduce(vcat, first.(results))), last.(results)
end







pc = PiecewiseConstantParameter(0.0:0.1:1.0, (rng, outdims) -> [randn(2) for _ in Base.OneTo(outdims)])
ps, st = LuxCore.setup(Random.default_rng(), pc)
pc(0.99, ps, st)

prob = ODEProblem((u,p,t)->-p .* u, rand(3), (0., 1.), rand(3))
pss = InitialConditions(prob, 2 => [0., 1.])
ps, st = LuxCore.setup(Random.default_rng(), pss)
pss(-0.1, ps, st)



@concrete terse struct InitialConditions <: LuxCore.AbstractLuxContainerLayer{(:lookup, :parameter)}
    name
    lookup
    parameter
end

function PiecewiseConstantParameter(timegrid::AbstractVector{<:Number}, default; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(searchsortedfirst, timegrid, cache)
    parameters = ParameterCollection(length(timegrid)+1, (rng, outdims) -> fill(default, outdims))
    PiecewiseConstantParameter(name, lookup, parameters)
end

function PiecewiseConstantParameter(timegrid::AbstractVector{<:Number}, default::Base.Callable; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(searchsortedfirst, timegrid, cache)
    parameters = ParameterCollection(length(timegrid) + 1, default)
    PiecewiseConstantParameter(name, lookup, parameters)
end

function (pc::PiecewiseConstantParameter)(t, ps, st)
    (; lookup, parameter) = pc 
    idx, lookup_st = LuxCore.apply(lookup, t, ps.lookup, st.lookup)
    param, parameter_st = LuxCore.apply(parameter, idx, ps.parameter, st.parameter)
    param, (; lookup = lookup_st, parameter = parameter_st)
end

