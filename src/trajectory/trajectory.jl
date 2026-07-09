"""
$(TYPEDEF)
The solution to an optimal control problem. 
# Fields
$(FIELDS)
# Note 
If present, `shooting_points` contains a list of `Tuple`s `(timeseries_index, last_shooting_point)`.  
"""
struct Trajectory{S, U, P, T, SH}
    "The symbolic system used for SymbolicIndexingInterface"
    sys::S
    "The state trajectory"
    u::U
    "The parameter values"
    p::P
    "The timepoints"
    t::T
    "The shooting values"
    shooting::SH
    "The shooting indices"
    shooting_indices::Vector{Int64}
end

SymbolicIndexingInterface.is_timeseries(::Type{<:Trajectory}) = Timeseries()
function SymbolicIndexingInterface.is_timeseries(
        ::Type{<:Trajectory{S, U, P, Nothing}}
    ) where {S, U, P}
    return NotTimeseries()
end
SymbolicIndexingInterface.symbolic_container(fp::Trajectory) = fp.sys
SymbolicIndexingInterface.state_values(fp::Trajectory) = fp.u
SymbolicIndexingInterface.parameter_values(fp::Trajectory) = fp.p
SymbolicIndexingInterface.current_time(fp::Trajectory) = fp.t

utype(traj::Trajectory) = eltype(first(traj.u))
ttype(traj::Trajectory) = eltype(traj.t)

is_shooting_solution(traj::Trajectory) = !isempty(traj.shooting)

shooting_violations(traj::Trajectory) = traj.shooting

function Base.getindex(T::Trajectory, ind::Symbol)
    if ind in keys(T.sys.variables)
        return vcat(getindex.(T.u, T.sys.variables[ind]))
    elseif ind in keys(T.sys.parameters)
        return getindex(T.p, T.sys.parameters[ind])
    end
    error(string("Invalid index: :", ind))
end
