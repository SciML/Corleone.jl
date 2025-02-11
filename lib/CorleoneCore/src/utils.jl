# Similar to LuxCore.update_state, but collects all time points within a given timespan 
function _collect_timepoints(st::NamedTuple, key::Symbol, tspan::Tuple{T, T}) where T
    timepoints = T[]
    collector = let timepoints = timepoints, key = key, tspan = tspan
        (kp, val) -> begin 
            if last(kp) == key && isa(val, AbstractVector)
                append!(timepoints, T[xi for xi in val if first(tspan) <= xi <= last(tspan)])
            end
            return val
        end
    end
    fmap_with_path(collector, st; exclude = LuxCore.Internal.isleaf)
    unique!(timepoints)
    sort!(timepoints)
    return timepoints
end

"""
$(FUNCTIONNAME)

Collect all `tstops` of the model given the time span.
Returns a sorted array with unique entries.
"""
function collect_tstops(st::NamedTuple, tspan::Tuple)
    _collect_timepoints(st, :tstops, tspan)
end

"""
$(FUNCTIONNAME)

Collect all `saveat` of the model given the time span.
Returns a sorted array with unique entries.
"""
function collect_saveat(st::NamedTuple, tspan::Tuple)
    _collect_timepoints(st, :saveat, tspan)
end

# Sorting and Copy

# Helper for time points
maybe_unique_sort(x) = x
maybe_unique_sort(x::AbstractVector) = sort(unique(x))
maybe_unique_sort!(x) = x
maybe_unique_sort!(x::AbstractVector) = sort!(unique!(x)) 

_maybecopy(x) = deepcopy(x)
_maybecopy(x::AbstractArray) = copy(x)

# Check for an AbstractTimeGridLayer
"""
$(FUNCTIONNAME)

Check if the structure `l` is a [`AbstractTimeGridLayer`](@ref) or a container of such a layer.
"""
function contains_timegrid_layer(l)
    return LuxCore.check_fmap_condition(Base.Fix2(isa, AbstractTimeGridLayer), AbstractTimeGridLayer, l)
end

"""
$(FUNCTIONNAME)

Check if the structure `l` is a [`AbstractTimeGridLayer`](@ref) which omits `tstops` or a container of such a layer.
"""
function contains_tstop_layer(l)
    return LuxCore.check_fmap_condition(Base.Fix2(isa, AbstractTimeGridLayer{true}), AbstractTimeGridLayer{true}, l)
end

"""
$(FUNCTIONNAME)

Check if the structure `l` is a [`AbstractTimeGridLayer`](@ref) which omits `saveat` or a container of such a layer.
"""
function contains_saveat_layer(l)
    return LuxCore.check_fmap_condition(Base.Fix2(isa, AbstractTimeGridLayer{<:Any, true}), AbstractTimeGridLayer{<:Any, true}, l)
end
