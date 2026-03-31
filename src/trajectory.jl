"""
$(TYPEDEF)
The solution to an optimal control problem. 
# Fields
$(FIELDS)
# Note 
If present, `shooting_points` contains a list of `Tuple`s `(timeseries_index, last_shooting_point)`.  
"""
struct Trajectory{S, U, P, T, C, SH, O}
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
    "Custom observed functions (NamedTuple of functions with signature (u, p, t) -> value)"
    custom_observed::O
    
    # Inner constructor for backward compatibility: default custom_observed to empty NamedTuple
    function Trajectory{S, U, P, T, C, SH}(sys::S, u::U, p::P, t::T, controls::C, shooting::SH) where {S, U, P, T, C, SH}
        new{S, U, P, T, C, SH, NamedTuple{(), Tuple{}}}(sys, u, p, t, controls, shooting, NamedTuple())
    end
    
    # Full constructor with all 7 fields
    function Trajectory{S, U, P, T, C, SH, O}(sys::S, u::U, p::P, t::T, controls::C, shooting::SH, custom_observed::O) where {S, U, P, T, C, SH, O}
        new{S, U, P, T, C, SH, O}(sys, u, p, t, controls, shooting, custom_observed)
    end
end

"""
$(SIGNATURES)

Extract plain symbol name from symbolic variable.
For MTK time-dependent variables like `u(t)`, extracts the base name `:u`.
For plain Symbols, returns the symbol unchanged.
Extended by CorleoneModelingToolkitExtension for MTK symbolic types.
"""
_maybesymbolifyme(x) = x

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

Return the names of the control parameters stored in `fp` as Symbols.
Normalizes MTK symbolic types (Num, BasicSymbolic) to Symbol for consistent comparison.
For time-dependent symbols like `u(t)`, extracts the base name `:u`.
"""
_control_names(fp::Trajectory) = isnothing(fp.controls) ? () : Tuple(_maybesymbolifyme(c.name) for c in values(fp.controls.model.controls))

"""
$(SIGNATURES)

Override parameter check: control parameter symbols are exposed as observed, not
as plain parameters, so `getsym`/`getp` route through `parameter_observed`.
Accepts both Symbol (`:u`) and MTK symbolic (`u(t)`) inputs.
"""
function SymbolicIndexingInterface.is_parameter(fp::Trajectory, sym)
    return is_parameter(fp.sys, sym) && !(_maybesymbolifyme(sym) in _control_names(fp))
end

"""
$(SIGNATURES)

Return `true` when `sym` is a control parameter of `fp`.
Accepts both Symbol (`:u`) and MTK symbolic (`u(t)`) inputs.
"""
function SymbolicIndexingInterface.is_observed(fp::Trajectory, sym)
    name = _maybesymbolifyme(sym)
    # Check control parameters and custom observed
    return name in _control_names(fp) || hasproperty(fp.custom_observed, name)
end

"""
$(SIGNATURES)

Return a time-dependent observed function for control parameter `sym`.
Used by `getsym` on timeseries objects to broadcast over all timepoints.
Accepts both Symbol (`:u`) and MTK symbolic (`u(t)`) inputs.
"""
function SymbolicIndexingInterface.observed(fp::Trajectory, sym)
    name = _maybesymbolifyme(sym)
    
    # Handle control parameters
    if name in _control_names(fp)
        return (u, p, t) -> getproperty(fp.controls(t), name)
    end
    
    # Handle custom observed functions
    if hasproperty(fp.custom_observed, name)
        return getproperty(fp.custom_observed, name)
    end
    
    error("Unknown observed symbol: $name")
end

"""
$(SIGNATURES)

Return a time-dependent parameter-observed function for control parameter `sym`.
The returned function has the signature `(p, t) -> value` and is used by `getp`.
For control parameters, the value is retrieved from the controls NamedTuple using
the symbol name (converted to Symbol for MTK compatibility).
Accepts both Symbol (`:u`) and MTK symbolic (`u(t)`) inputs.
"""
function SymbolicIndexingInterface.parameter_observed(fp::Trajectory, sym)
    # Convert MTK symbolic to Symbol for NamedTuple property access
    # _maybesymbolifyme extracts :u from u(t) or passes through plain :u
    name = _maybesymbolifyme(sym)
    return (p, t) -> begin
        if t isa AbstractVector
            map(ti -> getproperty(fp.controls(ti), name), t)
        else
            getproperty(fp.controls(t), name)
        end
    end
end

"""
$(SIGNATURES)

Convenience constructor for Trajectory with optional custom observed functions.
Custom observed functions can be passed as keyword arguments, e.g.:

    Trajectory(sys, u, p, t, controls, shooting; total_energy = (u, p, t) -> 0.5 * sum(u.^2))

When called without keyword arguments, uses an empty NamedTuple as default.
"""
function Trajectory(sys, u, p, t, controls, shooting; custom_observed...)
    return Trajectory(typeof(sys), typeof(u), typeof(p), typeof(t), typeof(controls), typeof(shooting), typeof(NamedTuple(custom_observed)))(sys, u, p, t, controls, shooting, NamedTuple(custom_observed))
end

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
