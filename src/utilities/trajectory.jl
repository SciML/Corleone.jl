"""
$(TYPEDEF)

Assemble saved segment solutions into a trajectory. Continuous handoffs occur once;
initial-condition boundaries retain both the incoming endpoint and outgoing start,
even when their states are equal. `reset_indices` indexes the outgoing samples.

Fields `u` and `t` contain the state samples and their times. `residuals` retains one
initial-condition residual vector per segment start. The trajectory stores `layer`
and its supplied Lux parameters `ps` and state `st`; parameter histories are evaluated
on demand through `layer.parameter`, rather than stored separately. State samples
are shared with the input solutions. Mutating `ps` or `st` changes subsequent parameter
queries without recomputing the saved states. Run the layer again after changing
parameters when both histories must describe the same prediction.
Gradients through state values, residuals, and parameter values are
supported on a fixed time grid; layout and symbolic metadata are not differentiated.
The symbolic cache is shared with `layer.sys` and exposed through
`SymbolicIndexingInterface.symbolic_container`. Query controls, quadrature states,
and bounds on that `ControlSymbolCache` wrapper.
No interpolation or integration is performed.

# Fields
$(FIELDS)
"""
@concrete terse struct Trajectory
    "The symbolic bookkeeping cache shared with the shooting layer."
    sys
    "Saved state samples, sharing their numerical arrays with the segment solutions."
    u
    "The actual saved solution times, including both sides of reset boundaries."
    t
    "State matching residual vectors per segment start; excludes on-demand parameter equalities."
    residuals
    "Indices in `u` and `t` of outgoing samples at initial-condition reset boundaries."
    reset_indices
    "The shooting layer used to evaluate parameter values on demand."
    layer
    "The supplied Lux parameters, retained by reference for parameter queries and AD."
    ps
    "The Lux state used for parameter queries, retained by reference for AD."
    st
end

# Keep construction separate from assembly: some downstream constructor rules
# dispatch too broadly on parametric type constructors.
"""
$(SIGNATURES)

Assemble saved segment solutions and residuals into a [`Trajectory`](@ref).
Remove duplicate samples at continuous handoffs while retaining both sides of
initial-condition resets. Saved state arrays, parameters, and Lux state are shared.
"""
Trajectory(layer, ps, st, solutions::AbstractVector, residuals) =
    build_trajectory(layer, ps, st, solutions, residuals)

# Use a constructor-specific rule so LuxCore's broad constructor rule cannot
# intercept the public five-argument API either.
function ChainRulesCore.rrule(
        config::RuleConfig{>:HasReverseMode}, ::Type{Trajectory},
        layer, ps, st, solutions::AbstractVector, residuals
    )
    return rrule_via_ad(config, build_trajectory, layer, ps, st, solutions, residuals)
end

# The layout is discrete metadata for a fixed time grid. State values never
# participate in deciding which samples survive a continuous handoff.
function trajectory_layout(layer, ps, st, solutions)
    reset_times = get_timestops(layer.initialcondition, ps.initialcondition, st.initialcondition, eltype(first(solutions).t))
    keep_start = [i == 1 || first(sol.t) in reset_times for (i, sol) in enumerate(solutions)]
    n = sum(length(sol.t) - !keep_start[i] for (i, sol) in enumerate(solutions))
    segments = Vector{Int}(undef, n)
    samples = Vector{Int}(undef, n)
    times = Vector{eltype(first(solutions).t)}(undef, n)
    reset_indices = Int[]
    k = 1
    for (i, sol) in enumerate(solutions)
        i > 1 && keep_start[i] && push!(reset_indices, k)
        for j in (firstindex(sol.t) + !keep_start[i]):lastindex(sol.t)
            segments[k] = i
            samples[k] = j
            times[k] = sol.t[j]
            k += 1
        end
    end
    return (; segments, samples, times, reset_indices)
end
@non_differentiable trajectory_layout(::Any, ::Any, ::Any, ::Any)

trajectory_symbolic_container(layer) = layer.sys
@non_differentiable trajectory_symbolic_container(::Any)

# Preallocate only the outer vector; the saved state arrays remain shared.
function gather_trajectory_states(states, segments, samples)
    u = similar(first(states), length(segments))
    for k in eachindex(segments)
        u[k] = states[segments[k]][samples[k]]
    end
    return u
end

function ChainRulesCore.rrule(::typeof(gather_trajectory_states), states, segments, samples)
    u = gather_trajectory_states(states, segments, samples)
    function gather_states_pullback(delta)
        delta = unthunk(delta)
        delta isa AbstractZero && return NoTangent(), ZeroTangent(), NoTangent(), NoTangent()
        # Keep a consistent nested array tangent even for unselected samples.
        # Accumulate rather than overwrite when a source sample occurs twice.
        dstates = map(us -> map(zero, us), states)
        for k in eachindex(segments)
            du = unthunk(delta[k])
            du isa AbstractZero && continue
            dstates[segments[k]][samples[k]] += du
        end
        return NoTangent(), ProjectTo(states)(dstates), NoTangent(), NoTangent()
    end
    return u, gather_states_pullback
end

function build_trajectory(layer, ps, st, solutions::AbstractVector, residuals)
    isempty(solutions) && throw(ArgumentError("Cannot construct a trajectory without segment solutions"))
    length(solutions) == length(residuals) || throw(DimensionMismatch("Expected one residual vector per segment"))
    layout = trajectory_layout(layer, ps, st, solutions)
    states = map(sol -> sol.u, solutions)
    u = gather_trajectory_states(states, layout.segments, layout.samples)
    return Trajectory(
        trajectory_symbolic_container(layer), u, layout.times,
        residuals, layout.reset_indices, layer, ps, st
    )
end

SymbolicIndexingInterface.is_timeseries(::Type{<:Trajectory}) = Timeseries()
SymbolicIndexingInterface.is_parameter_timeseries(::Type{<:Trajectory}) = Timeseries()
SymbolicIndexingInterface.symbolic_container(traj::Trajectory) = traj.sys
SymbolicIndexingInterface.state_values(traj::Trajectory) = traj.u
SymbolicIndexingInterface.current_time(traj::Trajectory) = traj.t

function SymbolicIndexingInterface.parameter_values(traj::Trajectory)
    return SymbolicIndexingInterface.parameter_values_at_time(traj, traj, last(traj.t))
end

function SymbolicIndexingInterface.parameter_values_at_time(::Trajectory, traj::Trajectory, t)
    first(traj.t) <= t <= last(traj.t) || throw(DomainError(t, "Time is outside the trajectory"))
    return first(traj.layer.parameter(t, traj.ps.parameter, traj.st.parameter))
end

function trajectory_parameter_times(traj)
    times = get_timestops(traj.layer.parameter, traj.ps.parameter, traj.st.parameter, eltype(traj.t))
    return sort!(unique!(vcat([first(traj.t)], filter(t -> first(traj.t) <= t <= last(traj.t), times))))
end
@non_differentiable trajectory_parameter_times(::Any)

function SymbolicIndexingInterface.get_parameter_timeseries_collection(traj::Trajectory)
    times = trajectory_parameter_times(traj)
    values = [first(traj.layer.parameter(t, traj.ps.parameter, traj.st.parameter)) for t in times]
    history = RecursiveArrayTools.DiffEqArray(values, times)
    return ParameterTimeseriesCollection((history,), parameter_values(traj))
end

SymbolicIndexingInterface.is_timeseries_parameter(traj::Trajectory, sym) = is_parameter(traj.sys, sym)
function SymbolicIndexingInterface.timeseries_parameter_index(traj::Trajectory, sym)
    is_parameter(traj.sys, sym) || return nothing
    return ParameterTimeseriesIndex(1, parameter_index(traj.sys, sym))
end
function SymbolicIndexingInterface.with_updated_parameter_timeseries_values(
        ::Trajectory, params, updates::Pair...
    )
    isempty(updates) && return params
    all(update -> first(update) == 1, updates) || throw(ArgumentError("Unknown parameter timeseries"))
    return last(last(updates))
end
# SciMLBase also dispatches on the parameter container. Resolve the intersection
# while retaining its wrapper when updating a trajectory's parameter timeseries.
@static if isdefined(SciMLBase, :DespecializedParameters)
    function SymbolicIndexingInterface.with_updated_parameter_timeseries_values(
            traj::Trajectory, params::SciMLBase.DespecializedParameters, updates::Pair...
        )
        updated = SymbolicIndexingInterface.with_updated_parameter_timeseries_values(traj, params.params, updates...)
        return SciMLBase.DespecializedParameters(updated)
    end
end

function SymbolicIndexingInterface.get_all_timeseries_indexes(traj::Trajectory, sym)
    is_parameter(traj.sys, sym) && return Set([1])
    return get_all_timeseries_indexes(traj.sys, sym)
end
function SymbolicIndexingInterface.get_all_timeseries_indexes(traj::Trajectory, syms::Union{AbstractArray, Tuple})
    return mapreduce(sym -> get_all_timeseries_indexes(traj, sym), union, syms; init = Set())
end
function SymbolicIndexingInterface.get_all_timeseries_indexes(traj::Trajectory, expr::Expr)
    return mapreduce(sym -> get_all_timeseries_indexes(traj, sym), union, expr.args; init = Set())
end

Base.getindex(traj::Trajectory, sym) = getsym(traj, sym)(traj)

utype(traj::Trajectory) = eltype(first(traj.u))
ttype(traj::Trajectory) = eltype(traj.t)
is_shooting_solution(traj::Trajectory) = !isempty(traj.reset_indices)
shooting_violations(traj::Trajectory) = traj.residuals

"""
$(SIGNATURES)

Return scalar state matching residuals followed by parameter matching residuals.
State residuals use chronological segment order, with variables in canonical
state index order within each segment. The
values are assigned states minus incoming states, as stored in
`trajectory.residuals`. Parameter residuals are computed on demand from the stored
layer, parameters, and state. Each inserted right-hand interval index `i` contributes
`values[i] - values[i - 1]`, in canonical parameter order and then increasing index.
Original parameter switches and fixed assignments do not contribute residuals.

These are equality-constraint values; matching constraint bounds are zero.
Numerical derivatives flow through the residual values. Use [`shooting_constraints!`](@ref)
to write directly into an existing optimizer buffer.
"""
function shooting_constraints(trajectory::Trajectory)
    residuals = isempty(trajectory.residuals) ? similar(first(trajectory.u), 0) : reduce(vcat, trajectory.residuals)
    return append_parameter_constraints(trajectory.layer.parameter, trajectory.ps.parameter, trajectory.st.parameter, residuals)
end

"""
$(SIGNATURES)

Write [`shooting_constraints`](@ref) into `res` in the same order and return `res`.
The buffer length must equal the total number of scalar residuals; otherwise throw
`DimensionMismatch` before writing. This avoids allocating an intermediate flattened
vector. The buffer's element type must support the residual values, including AD
number types. Backends that do not support array mutation should use the allocating
`shooting_constraints` form instead.
"""
function shooting_constraints!(res::AbstractVector, trajectory::Trajectory)
    expected = sum(length, trajectory.residuals; init = 0) +
        number_of_parameter_constraints(trajectory.layer.parameter, trajectory.st.parameter)
    length(res) == expected || throw(
        DimensionMismatch(
            "Expected a shooting-constraint buffer of length $expected, got $(length(res))."
        )
    )
    index = firstindex(res)
    for residual in trajectory.residuals
        for value in residual
            res[index] = value
            index += 1
        end
    end
    write_parameter_constraints!(res, index, trajectory.layer.parameter, trajectory.ps.parameter, trajectory.st.parameter)
    return res
end
