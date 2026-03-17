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

"""
$(TYPEDEF)

Time-aligned control signal values.

# Fields
$(FIELDS)
"""
struct ControlSignal{T, U}
    t::Vector{T}
    u::Vector{U}
end

# Must be a timeseries object, and implement `current_time` and `state_values`
"""
$(SIGNATURES)

Declare [`ControlSignal`](@ref) as a symbolic time series.
"""
SymbolicIndexingInterface.is_timeseries(::Type{<:ControlSignal}) = Timeseries()

"""
$(SIGNATURES)

Return time coordinates of `a`.
"""
SymbolicIndexingInterface.current_time(a::ControlSignal) = a.t

"""
$(SIGNATURES)

Return signal values of `a`.
"""
SymbolicIndexingInterface.state_values(a::ControlSignal) = a.u


"""
$(SIGNATURES)

Declare [`Trajectory`](@ref) as a symbolic time series.
"""
SymbolicIndexingInterface.is_timeseries(::Type{<:Trajectory}) = SymbolicIndexingInterface.Timeseries()

"""
$(SIGNATURES)

Return symbolic container associated with `fp`.
"""
SymbolicIndexingInterface.symbolic_container(fp::Trajectory) = fp.sys

"""
$(SIGNATURES)

Return state values of `fp`.
"""
SymbolicIndexingInterface.state_values(fp::Trajectory) = fp.u

"""
$(SIGNATURES)

Return parameter values of `fp`.
"""
SymbolicIndexingInterface.parameter_values(fp::Trajectory) = fp.p

"""
$(SIGNATURES)

Return time coordinates of `fp`.
"""
SymbolicIndexingInterface.current_time(fp::Trajectory) = fp.t

"""
$(SIGNATURES)

Return control parameter time-series collection of `fs`.
"""
SymbolicIndexingInterface.get_parameter_timeseries_collection(fs::Trajectory) = fs.controls

"""
$(SIGNATURES)

Declare trajectory parameters as time-series values.
"""
SymbolicIndexingInterface.is_parameter_timeseries(::Type{<:Trajectory}) = SymbolicIndexingInterface.Timeseries()

"""
$(SIGNATURES)

Return the element type of state vectors in `traj`.
"""
utype(traj::Trajectory) = eltype(first(traj.u))

"""
$(SIGNATURES)

Return the scalar time type of `traj`.
"""
ttype(traj::Trajectory) = eltype(traj.t)

"""
$(SIGNATURES)

Return `true` if `traj` contains shooting continuity data.
"""
is_shooting_solution(traj::Trajectory) = !isnothing(traj.shooting) && !isempty(traj.shooting)

"""
$(SIGNATURES)

Return stored shooting continuity violations.
"""
shooting_violations(traj::Trajectory) = traj.shooting

"""
$(SIGNATURES)

Return symbolic values indexed by `sym` from `A`.

Parameter indexing through `getindex` is deprecated; use `A.ps[sym]` instead.
"""
function Base.getindex(A::Trajectory, sym)
    if is_parameter(A, sym)
        error("Indexing with parameters is deprecated. Use `sol.ps[$sym]` for parameter indexing.")
    end
    return getsym(A, sym)(A)
end

"""
$(SIGNATURES)

Expose parameter indexing proxy as `traj.ps`.
"""
function Base.getproperty(fs::Trajectory, s::Symbol)
    return s === :ps ? ParameterIndexingProxy(fs) : getfield(fs, s)
end

function shooting_constraints!(res, traj)
    (; shooting) = traj
    isnothing(shooting) && return res
    offset = 0
    for xi in fleaves(shooting), xij in xi
        offset += 1
        res[offset] = xij
    end
    return res
end

function shooting_constraints(traj)
    (; shooting) = traj
    isnothing(shooting) && return eltype(traj.u[1])[]
    return vcat(fleaves(shooting)...)
end
