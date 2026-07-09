"""
$(TYPEDEF)

A `ShootingSegment` is a single shooting interval composed of one or more
piecewise-constant `ControlSegment`s. The segments are chained end-to-end, and the
aggregation logic trims duplicated boundary points when concatenating across the
contained `ControlSegment`s.
"""
struct ShootingSegment{N, C <: ControlSegment, S} <: AbstractCompositeSolution{N, C}
    segments::NTuple{N,C}
    sys::S
end
