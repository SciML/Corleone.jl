# TODO Maybe it is a good idea to rename "timepoints" to knots and allow more general versions here (e.g. for u[1] ...)

"""
$(TYPEDEF)

An approximation of the unknown control signal using a piecewise constant signal 
over time. 

# Fields 
$(FIELDS)
"""
struct PiecewiseConstant{T<:AbstractVector} <: AbstractLuxLayer
    "The timepoints of the approximation"
    timepoints::T
end

function PiecewiseConstant(c::PiecewiseConstant, lo, hi)
    idx = findall(lo .<= c.timepoints .< hi)
    return PiecewiseConstant(c.timepoints[idx])
end

LuxCore.parameterlength(x::PiecewiseConstant) = length(x.timepoints)
LuxCore.statelength(x::PiecewiseConstant) = 4 + length(x.timepoints)

LuxCore.initialstates(::Random.AbstractRNG, x::PiecewiseConstant) = (; timepoints=copy(x.timepoints), method=Val{:searchsorted}(), guess=1, indexset = eachindex(x.timepoints), min_index=firstindex(x.timepoints), max_index=lastindex(x.timepoints),)
LuxCore.initialparameters(::Random.AbstractRNG, x::PiecewiseConstant) = (; local_controls=zeros(LuxCore.parameterlength(x)))

__getindex(x::AbstractArray{<:Any,N}, idx) where {N} = selectdim(x, N, idx)

function __search_index(::Any, timepoints, t, guess)
    searchsortedlast(timepoints, t + eps()) 
end

function __search_index(::Val{:correlated}, timepoints, t, guess)
    searchsortedlastcorrelated(timepoints, t + eps(), guess) 
end

function (x::PiecewiseConstant)(args::Tuple, ps, st::NamedTuple)
    (; timepoints, method, guess, min_index, max_index, ) = st
    #@info timepoints
    (; local_controls) = ps
    t = Base.last(args)
    idx = min(max(__search_index(method, timepoints, t, guess), firstindex(timepoints)), lastindex(timepoints))
    __getindex(local_controls, idx), merge(st, (; guess=idx))
end


