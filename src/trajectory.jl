"""
$(TYPEDEF)
The solution to an optimal control problem. 
# Fields
$(FIELDS)
# Note 
If present, `shooting_points` contains a list of `Tuple`s `(timeseries_index, last_shooting_point)`.  
"""
struct Trajectory{S, U, P, T, C, SH, CT}
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
    "Pre-computed control timeseries values for SII parameter-timeseries indexing"
    control_timeseries::CT
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

Check whether `sym` is a timeseries parameter in `fp`.
Control parameters are timeseries parameters — their values change over time.
This routes `getp`/`getsym` through SII's `GetParameterTimeseriesIndex` pathway
for discrete-time indexed access, rather than the old `GetParameterObserved` pathway.
Accepts both Symbol (`:u`) and MTK symbolic (`u(t)`) inputs.
"""
function SymbolicIndexingInterface.is_timeseries_parameter(fp::Trajectory, sym)
    return _maybesymbolifyme(sym) in _control_names(fp)
end

"""
$(SIGNATURES)

Return `true` when `sym` is an observed (algebraic/postprocessing) quantity of `fp`.
Control parameters are no longer classified as observed — they are timeseries
parameters. This frees the `is_observed` channel for genuine algebraic observables
(e.g. postprocessing layers, MTK observed equations).
Accepts both Symbol (`:u`) and MTK symbolic (`u(t)`) inputs.
"""
function SymbolicIndexingInterface.is_observed(fp::Trajectory, sym)
    return false
end

"""
$(SIGNATURES)

Return the `ParameterTimeseriesIndex` for control parameter `sym` in `fp`.
This identifies which timeseries object in the `ParameterTimeseriesCollection`
corresponds to this control, enabling SII's `GetParameterTimeseriesIndex` pathway.
Accepts both Symbol (`:u`) and MTK symbolic (`u(t)`) inputs.
"""
function SymbolicIndexingInterface.timeseries_parameter_index(fp::Trajectory, sym)
    if is_timeseries_parameter(fp, sym)
        name = _maybesymbolifyme(sym)
        ctrl_names = collect(_control_names(fp))
        ts_idx = findfirst(==(name), ctrl_names)
        return ParameterTimeseriesIndex(ts_idx, 1)
    end
    return nothing
end

"""
$(SIGNATURES)

Return a time-dependent parameter-observed function for control parameter `sym`.
The returned function has the signature `(p, t) -> value` and supports
interpolated evaluation at arbitrary time points (ZOH by default).

Note: This is a standalone evaluation API, not triggered through `getp` dispatch.
For `getp`/`getsym` access, controls now route through the `GetParameterTimeseriesIndex`
pathway, which provides discrete-time indexed access via the `ParameterTimeseriesCollection`.
`parameter_observed` remains available for continuous-time interpolated evaluation
(e.g. in objective/constraint functions at arbitrary time points).

Accepts both Symbol (`:u`) and MTK symbolic (`u(t)`) inputs.
"""
function SymbolicIndexingInterface.parameter_observed(fp::Trajectory, sym)
    if is_timeseries_parameter(fp, sym)
        name = _maybesymbolifyme(sym)
        ctrl_eval = fp.controls  # capture only the control evaluator, not the full Trajectory
        return (p, t) -> begin
            if t isa AbstractVector
                map(ti -> getproperty(ctrl_eval(ti), name), t)
            else
                getproperty(ctrl_eval(t), name)
            end
        end
    end
    return nothing
end

"""
$(SIGNATURES)

Declare [`Trajectory`](@ref) as containing parameter timeseries data.
This is required for SII's `GetParameterTimeseriesIndex` dispatch.
"""
SymbolicIndexingInterface.is_parameter_timeseries(::Type{<:Trajectory}) = Timeseries()

"""
$(SIGNATURES)

Return the `ParameterTimeseriesCollection` for `fp`, containing pre-computed control
values at all trajectory timepoints. Each control is stored as a `ControlSignal`
(timeseries object), enabling SII's `GetParameterTimeseriesIndex` to extract
control values at discrete time indices.
"""
function SymbolicIndexingInterface.get_parameter_timeseries_collection(fp::Trajectory)
    return fp.control_timeseries
end

"""
$(SIGNATURES)

Build the `ParameterTimeseriesCollection` by evaluating each control at all
corresponding trajectory timepoints. The control evaluator (`StatefulLuxLayer`)
is called at each time to produce a `ControlSignal` for each control name.
"""
function _build_control_timeseries(controls, t, p)
    ctrl_names = Tuple(_maybesymbolifyme(c.name) for c in values(controls.model.controls))
    ctrl_signals = NamedTuple{ctrl_names}(map(ctrl_names) do name
        vals = [getproperty(controls(ti), name) for ti in t]
        ControlSignal(collect(t), vals)
    end)
    return ParameterTimeseriesCollection(ctrl_signals, p)
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
    if is_parameter(A, sym) && !is_timeseries_parameter(A, sym)
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
