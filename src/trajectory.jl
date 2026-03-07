"""
$(TYPEDEF)
The solution to an optimal control problem. 
# Fields
$(FIELDS)
# Note 
If present, `shooting_points` contains a list of `Tuple`s `(timeseries_index, last_shooting_point)`.  
"""
struct Trajectory{S, U, P, T, C, SH}
    "The symbolic system used for SymbolicIndexingInterface"
    sys::S
    "The state trajectory"
    u::U
    "The parameter values"
    p::P
    "The timepoints"
    t::T
    "The control signals"
    controls::C
    "The shooting values"
    shooting::SH
end

struct ControlSignal{T, U}
  t::T
  u::U
end

# Must be a timeseries object, and implement `current_time` and `state_values`
SymbolicIndexingInterface.is_timeseries(::Type{<:ControlSignal}) = Timeseries()
SymbolicIndexingInterface.current_time(a::ControlSignal) = a.t
SymbolicIndexingInterface.state_values(a::ControlSignal) = a.u


SymbolicIndexingInterface.is_timeseries(::Type{<:Trajectory}) = SymbolicIndexingInterface.Timeseries()
SymbolicIndexingInterface.symbolic_container(fp::Trajectory) = fp.sys
SymbolicIndexingInterface.state_values(fp::Trajectory) = fp.u
SymbolicIndexingInterface.parameter_values(fp::Trajectory) = fp.p
SymbolicIndexingInterface.current_time(fp::Trajectory) = fp.t
SymbolicIndexingInterface.get_parameter_timeseries_collection(fs::Trajectory) = fs.controls
SymbolicIndexingInterface.is_parameter_timeseries(::Type{<:Trajectory}) = SymbolicIndexingInterface.Timeseries()

utype(traj::Trajectory) = eltype(first(traj.u))
ttype(traj::Trajectory) = eltype(traj.t)

is_shooting_solution(traj::Trajectory) = !isempty(traj.shooting)

shooting_violations(traj::Trajectory) = traj.shooting

function Base.getindex(A::Trajectory, sym)
    if is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `sol.ps[$sym]` for parameter indexing.")
    end
    return getsym(A, sym)(A)
end

function Base.getproperty(fs::Trajectory, s::Symbol)
    s === :ps ? ParameterIndexingProxy(fs) : getfield(fs, s)
end
