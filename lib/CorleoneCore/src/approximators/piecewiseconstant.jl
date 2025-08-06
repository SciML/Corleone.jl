# TODO Maybe it is a good idea to rename "timepoints" to knots and allow more general versions here (e.g. for u[1] ...)

"""
$(TYPEDEF)

An approximation of the unknown control signal using a piecewise constant signal 
over time. 

# Fields 
$(FIELDNAMES)
"""
struct PiecewiseConstant{T<:AbstractVector} <: AbstractLuxLayer
    "The timepoints of the approximation"
    timepoints::T
end

__getindex(x::AbstractArray{<:Any,N}, idx) where {N} = selectdim(x, N, idx)
has_tstops(::PiecewiseConstant) = true
get_tstops(x::PiecewiseConstant) = getfield(x, :timepoints)

LuxCore.parameterlength(x::PiecewiseConstant) = length(x.timepoints)
LuxCore.statelength(x::PiecewiseConstant) = 4

LuxCore.initialstates(::Random.AbstractRNG, x::PiecewiseConstant) = (; method=Val{:searchsorted}(), guess=1, min_index=firstindex(x.timepoints), max_index=lastindex(x.timepoints))
LuxCore.initialparameters(::Random.AbstractRNG, x::PiecewiseConstant) = (; local_controls=collect(LinRange(0.0, 1.0, LuxCore.parameterlength(x))))

function __search_index(::Any, timepoints, t, guess)
    searchsortedfirst(timepoints, t) - 1
end

function __search_index(::Val{:correlated}, timepoints, t, guess)
    searchsortedfirstcorrelated(timepoints, t, guess) - 1
end

function (x::PiecewiseConstant)(args::Tuple, ps, st::NamedTuple)
    (; method, guess, min_index, max_index) = st
    (; timepoints) = x
    (; local_controls) = ps
    t = Base.last(args)
    idx = clamp(__search_index(method, timepoints, t, guess), min_index, max_index)
    __getindex(local_controls, idx), (; method, guess=idx, min_index, max_index)
end


