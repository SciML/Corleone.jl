"""
$(TYPEDEF)

A `Trajectory` is a collection of `ShootingSegment`s that together represent the
solution to a `DEProblem` over a time interval. Each shooting segment corresponds
to an independent portion of the trajectory. If there is only one segment, the
trajectory is simply that shooting segment. If there are multiple segments, the
trajectory represents a multiple-shooting solution, and continuity between
shooting segments is enforced by `shooting_constraints`.

# Fields
$(FIELDS)

# Note
The trajectory defines convenience accessors for the state, control, and parameter
values across all segments, as well as the current time by overloading
`getproperty` and `getindex`.

```julia
traj.u          # state values across all segments (with controls appended)
traj.u_minimal  # raw ODE state values across all segments
traj.c          # control values across all control intervals
traj.p          # parameter values of the first segment
traj.t          # current time across all segments
```
"""
struct Trajectory{N, S <: ShootingSegment, SYS} <: AbstractCompositeSolution{S}
    "The shooting segments of the trajectory"
    segments::NTuple{N, S}
    "The symbolic system cache"
    sys::SYS
end

SymbolicIndexingInterface.symbolic_container(traj::Trajectory) = traj.sys

SymbolicIndexingInterface.state_values(traj::Trajectory) = _state_values(traj, Colon())
SymbolicIndexingInterface.state_values(traj::Trajectory, idxs) = _state_values(traj, idxs)
# Disambiguates against Solutions.state_values(::AbstractCompositeSolution, ::Colon).
SymbolicIndexingInterface.state_values(traj::Trajectory, ::Colon) = _state_values(traj, Colon())

function _state_values(traj::Trajectory{N}, idxs = Colon()) where {N}
    q_idxs = isa(idxs, Colon) ? quadrature_indices(traj.sys) : intersect(quadrature_indices(traj.sys), idxs)
    isempty(q_idxs) && return _aggregate_trim(Base.Fix2(state_values, idxs), traj)
    segs = map(Base.Fix2(state_values, idxs), traj.segments)
    offset = zero(first(first(segs)))
    selector = if isa(idxs, Colon)
        [i ∈ q_idxs for i in eachindex(offset)]
    else
        [i ∈ q_idxs for i in idxs]
    end
    return reduce(vcat, __accumulate_quadratures(offset, selector, segs))
end

function __accumulate_quadratures(offset, selector, segs::Tuple)
    current = first(segs)
    seg = map(Base.Fix1(+, offset), current)
    offset = selector .* last(seg)
    return (seg[1:(end - 1)], __accumulate_quadratures(offset, selector, Base.tail(segs))...)
end

function __accumulate_quadratures(offset, selector, segs::Tuple{T}) where {T}
    current = first(segs)
    seg = map(Base.Fix1(+, offset), current)
    return (seg,)
end

function Base.getproperty(traj::Trajectory, sym::Symbol)
    if sym == :ps
        return ParameterIndexingProxy(traj)
    elseif sym == :u
        return state_values(traj)
    elseif sym == :c
        return control_values(traj)
    elseif sym == :t
        return current_time(traj)
    elseif sym == :u_minimal
        return minimal_state_values(traj)
    elseif sym == :p
        return parameter_values(traj)
    else
        return getfield(traj, sym)
    end
end

function Base.getindex(traj::Trajectory, i::Int)
    return traj.u[i]
end

Base.Matrix(traj::Trajectory) = hcat(state_values(traj)...)

function Base.getindex(traj::Trajectory, i)
    index = variable_index(traj, i)
    if index !== nothing
        return getindex.(state_values(traj), index)
    end
    return eltype(first(first(traj.segments).segments).u)[]
end

n_shooting_constraints(::Trajectory{1}) = 0 
n_shooting_constraints(t::Trajectory{N}) where N = sum(t.segments) do seg
    n_shooting_constraints(seg)
end 

function shooting_constraints(trajectory::Trajectory{1})
    return eltype(first(first(trajectory.segments).segments).sol.u[1])[]
end

function shooting_constraints!(res, ::Trajectory{1})
    return res
end

function shooting_constraints(trajectory::Trajectory{N}) where {N}
    return reduce(vcat, map(zip(Base.front(trajectory.segments), Base.tail(trajectory.segments))) do (a,b)
        shooting_constraints(a, b)
    end)
end

function shooting_constraints!(res::AbstractVector, trajectory::Trajectory)
    offset = 0 
    (; segments) = trajectory
    foreach(zip(Base.front(segments), Base.tail(segments))) do (a, b)
        N = n_shooting_constraints(b)
        @views shooting_constraints!(res[offset .+ (1:N)], a, b)
        offset += N
    end
    return res
end
