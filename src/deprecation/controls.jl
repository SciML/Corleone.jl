"""
$(SIGNATURES)

Deprecated constructor for a piecewise-constant control. Return a
`Tunable(name; kind = Val(:parameter), ...) => switching_times` definition.
Legacy `t` contains the start of each control interval, so the first entry is
removed from the returned switching grid. `controls` may be a vector, scalar, or
legacy callback `(rng, times, bounds) -> values`. Callbacks run during Lux setup.

Scalar bounds and uniform bound vectors are supported. Nonuniform per-time bounds
cannot be represented by the current annotation bounds and are rejected. Pass the
returned definition directly to `ShootingLayer`, or use `index => definition` in a
deprecated shooting constructor to select a parameter independently of `name`.
"""
function ControlParameter(t::AbstractVector; name::Symbol = gensym(:w),
        controls = nothing, bounds = (-Inf, Inf))
    Base.depwarn("ControlParameter is deprecated; use Tunable(parameter; default, lowerbound, upperbound) => switching_times. The new grid contains interval boundaries, excluding the first interval start.", :ControlParameter)
    isempty(t) && throw(ArgumentError("A legacy control grid must contain at least one interval start."))
    issorted(t) && allunique(t) || throw(ArgumentError("Control times must be sorted and unique."))
    limits = bounds isa Tuple ? bounds : bounds(t)
    lower, upper = legacy_bounds(limits, length(t), "control")
    all(==(first(lower)), lower) && all(==(first(upper)), upper) ||
        throw(ArgumentError("Nonuniform per-time control bounds cannot be translated to scalar annotation bounds. Use the new API and construct optimizer bounds explicitly."))
    controls isa AbstractVector && length(controls) != length(t) &&
        throw(DimensionMismatch("Expected $(length(t)) legacy control values."))
    values = isnothing(controls) ? zero(float(first(t))) : controls
    initialize = function (rng, times::AbstractVector)
        indices = [max(firstindex(t), searchsortedlast(t, time)) for time in times]
        selected_times = t[indices]
        selected_bounds = (lower[indices], upper[indices])
        out = if values isa Number
            fill(values, length(times))
        elseif values isa AbstractVector
            values[indices]
        else
            values(rng, selected_times, selected_bounds)
        end
        return out
    end
    return Tunable(name; kind = Val(:parameter), default = initialize,
        lowerbound = first(lower), upperbound = first(upper)) => t[2:end]
end

function legacy_bounds(bounds, n, description)
    bounds isa Tuple && length(bounds) == 2 ||
        throw(ArgumentError("$description bounds must be a (lower, upper) tuple."))
    lower, upper = map(bounds) do side
        side isa Number ? fill(side, n) : side
    end
    lower isa AbstractVector && upper isa AbstractVector && length(lower) == length(upper) == n ||
        throw(ArgumentError("Inconsistent $description bounds; expected $n entries."))
    all(lower .<= upper) || throw(ArgumentError("Inconsistent $description bounds."))
    return lower, upper
end
