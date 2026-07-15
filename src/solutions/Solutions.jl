using Reexport

@reexport module Solutions
using SymbolicIndexingInterface
using SciMLBase
using ConcreteStructs
using DocStringExtensions

abstract type AbstractCompositeSolution{T} end

function _aggregate_trim(f, seg::AbstractCompositeSolution)
    (; segments) = seg
    N = length(segments)
    pieces = map(enumerate(segments)) do (i, s)
        vals = f(s)
        (i == N) ? vals : vals[begin:(end - 1)]
    end
    return reduce(vcat, pieces)
end

get_utype(seg::AbstractCompositeSolution) = get_utype(first(seg.segments))

SymbolicIndexingInterface.symbolic_container(seg::AbstractCompositeSolution) = seg.sys
SymbolicIndexingInterface.state_values(seg::AbstractCompositeSolution) = _aggregate_trim(state_values, seg)
minimal_state_values(seg::AbstractCompositeSolution) = _aggregate_trim(minimal_state_values, seg)
SymbolicIndexingInterface.current_time(seg::AbstractCompositeSolution) = _aggregate_trim(current_time, seg)

control_values(seg::AbstractCompositeSolution) = collect(map(control_values, seg.segments))
control_values(seg::AbstractCompositeSolution{<:AbstractCompositeSolution}) = reduce(vcat, map(control_values, seg.segments))

SymbolicIndexingInterface.parameter_values(seg::AbstractCompositeSolution) = parameter_values(first(seg.segments))

include("cache.jl")
include("control_segment.jl")
include("shooting_segment.jl")
include("trajectory.jl")

export ControlSymbolCache
export Trajectory, shooting_constraints, shooting_constraints!
export minimal_state_values, control_values

end
