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

function SymbolicIndexingInterface.state_values(traj::Trajectory)
    q_idxs = quadrature_indices(traj.sys)
    n_quad = length(q_idxs)
    segs = traj.segments
    T = get_utype(traj)
    cumulative_offset = zeros(T, n_quad)
    out = Vector{Vector{T}}()
    for (i, seg) in enumerate(segs)
        vals = map(copy, state_values(seg))
        terminal_quad = n_quad > 0 ? last(vals)[q_idxs] : T[]
        for v in vals
            v[q_idxs] .+= cumulative_offset
        end
        if i < length(segs)
            vals = vals[begin:(end - 1)]
        end
        cumulative_offset = cumulative_offset .+ terminal_quad
        append!(out, vals)
    end
    return out
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

function shooting_constraints(trajectory::Trajectory{1})
    return eltype(first(first(trajectory.segments).segments).sol.u[1])[]
end

function shooting_constraints(trajectory::Trajectory{N}) where {N}
    T = eltype(first(first(trajectory.segments).segments).sol.u[1])
    n_states = length(first(minimal_state_values(first(trajectory.segments))))
    res = Vector{T}(undef, (N - 1) * n_states)
    shooting_constraints!(res, trajectory)
    return res
end

function shooting_constraints!(res::AbstractVector, trajectory::Trajectory{N}) where {N}
    N == 1 && return res
    n_states = length(first(minimal_state_values(first(trajectory.segments))))
    for i in 2:N
        prev_end = minimal_state_values(trajectory.segments[i - 1])[end]
        curr_start = minimal_state_values(trajectory.segments[i])[begin]
        offset = (i - 2) * n_states
        for j in 1:n_states
            res[offset + j] = curr_start[j] - prev_end[j]
        end
    end
    return res
end
