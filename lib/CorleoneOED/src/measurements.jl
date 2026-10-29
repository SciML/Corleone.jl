"""
$(TYPEDEF)

A measurement specification with an identifier, schedule, and observation map
`observed(u, p, t)`. A map may return a scalar or a vector. All outputs of one
measurement share its weight. Measurement weights are tunable parameters in `[0, 1]`.
"""
abstract type AbstractMeasurement end

"""
$(TYPEDEF)

Observe at the exact, sorted `tpoints`. There is one weight per point, including
any point at the final time. Information is linear in the weights: `w * J'J`.

# Fields
$(FIELDS)
"""
struct DiscreteMeasurement{T, F} <: AbstractMeasurement
    "The unique parameter symbol for this measurement's weights."
    id::Symbol
    "The observation times, including endpoints when requested."
    tpoints::T
    "The observation map `(u, p, t) -> y`."
    observed::F
end

"""
$(TYPEDEF)

Integrate information with a piecewise-constant measurement density. `tpoints`
contains switching times; optional problem endpoints are removed when constructing
intervals. An empty grid gives one density over the whole time span.

# Fields
$(FIELDS)
"""
struct ContinuousMeasurement{T, F} <: AbstractMeasurement
    "The unique parameter symbol for this measurement's densities."
    id::Symbol
    "The sorted density schedule."
    tpoints::T
    "The observation map `(u, p, t) -> y`."
    observed::F
end

observation_values(value::Number) = [value]
observation_values(value) = vec(collect(value))

function measurement_schedule(m::M, tspan) where {M <: AbstractMeasurement}
    times = collect(promote_type(map(typeof, tspan)...), m.tpoints)
    issorted(times) && allunique(times) || throw(ArgumentError("Measurement times must be sorted and unique."))
    all(t -> isfinite(t) && first(tspan) <= t <= last(tspan), times) ||
        throw(ArgumentError("Measurement times must lie within the problem time span."))
    m isa DiscreteMeasurement && isempty(times) && throw(ArgumentError("A discrete measurement needs at least one time."))
    constructor = m isa DiscreteMeasurement ? DiscreteMeasurement : ContinuousMeasurement
    return constructor(m.id, times, m.observed)
end
measurement_switches(m::DiscreteMeasurement, tspan) = m.tpoints[2:end]
measurement_switches(m::ContinuousMeasurement, tspan) = filter(t -> first(tspan) < t < last(tspan), m.tpoints)
