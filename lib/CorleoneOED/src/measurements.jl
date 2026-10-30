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
any points at the initial and final time. `tpoints` must be nonempty, finite,
unique, and within the problem span; validation occurs in [`OEDLayer`](@ref).
Information is linear in the weights: `w * J'J`, with one shared weight for every
output row of `observed(u, p, t)`. Identifiers must be unique and distinct from
model symbols. Observation maps must accept symbolic inputs with a fixed output
size.

Events are added to the saved grid. At a shooting reset, discrete information
uses the last saved sample at that time, the outgoing/post-reset state. Discrete
budgets sum event weights; weights between events do not create extra observations.

# Fields
$(FIELDS)
"""
struct DiscreteMeasurement{T, F} <: AbstractMeasurement
    "The unique parameter symbol for this measurement's weights."
    id::Symbol
    "Sorted actual observation events; includes initial/final times only when supplied."
    tpoints::T
    "The symbolic-compatible map `(u, p, t) -> y`, with fixed scalar/vector output size."
    observed::F
end

"""
$(TYPEDEF)

Integrate information with a piecewise-constant measurement density. `tpoints`
contains switching times; optional problem endpoints are removed when constructing
intervals. There is one value per resulting interval, not per supplied point.
An empty grid gives one density over the whole time span. Times must be finite,
sorted, unique, and inside the problem span; validation occurs in [`OEDLayer`](@ref).

At a switch, the density is right-continuous. All output rows of `observed(u, p, t)`
share this density, and information is weighted linearly. Budgets integrate density
by interval duration. Identifiers must be unique and distinct from model symbols;
observation maps must accept symbolic inputs with a fixed scalar/vector output size.

# Fields
$(FIELDS)
"""
struct ContinuousMeasurement{T, F} <: AbstractMeasurement
    "The unique parameter symbol for this measurement's densities."
    id::Symbol
    "Sorted density-switch times; optional problem endpoints are only interval boundaries."
    tpoints::T
    "The symbolic-compatible map `(u, p, t) -> y`, with fixed scalar/vector output size."
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
