"""
$(TYPEDEF)

A `ShootingSegment` is a single shooting interval composed of one or more
piecewise-constant `ControlSegment`s. The segments are chained end-to-end, and the
aggregation logic trims duplicated boundary points when concatenating across the
contained `ControlSegment`s.
"""
struct ShootingSegment{C, T <: Base.AbstractVecOrTuple{<:ControlSegment}, S, L} <: AbstractCompositeSolution{C}
    segments::T
    sys::S
    shooting_variables::L
end

function ShootingSegment(segments::T, sys::S, shooting_variables::L) where {T, S, L}
    return ShootingSegment{eltype(segments), T, S, L}(segments, sys, shooting_variables)
end

function shooting_constraints(a::ShootingSegment, b::ShootingSegment)
    getter = map(Base.Fix1(variable_index, b.sys), b.shooting_variables)
    last_state_values(a.segments[end])[getter] .- first_state_values(b.segments[1])[getter]
end

function shooting_constraints!(res, a::ShootingSegment, b::ShootingSegment)
    getter = map(Base.Fix1(variable_index, b.sys), b.shooting_variables)
    res .=     last_state_values(a.segments[end])[getter] .- first_state_values(b.segments[1])[getter]
end

n_shooting_constraints(a::ShootingSegment) = size(a.shooting_variables, 1)