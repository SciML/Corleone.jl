"""
$(TYPEDEF)

A `ShootingSegment` is a single shooting interval composed of one or more
piecewise-constant `ControlSegment`s. The segments are chained end-to-end, and the
aggregation logic trims duplicated boundary points when concatenating across the
contained `ControlSegment`s.
"""
struct ShootingSegment{C, T <: Base.AbstractVecOrTuple{<:ControlSegment}, S} <: AbstractCompositeSolution{C}
    segments::T
    sys::S
end

function ShootingSegment(segments::T, sys::S) where {T, S}
    return ShootingSegment{eltype(segments), T, S}(segments, sys)
end
