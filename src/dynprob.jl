"""
$(TYPEDEF)

A prepared dynamic optimization model for a [`ShootingLayer`](@ref). Construct with
`CorleoneDynamicOptProblem(layer, loss, constraints...; ps, st, rng)`, or use the
keyword form `CorleoneDynamicOptProblem(layer; loss, constraints, ps, st, rng)`.

`loss` is a scalar trajectory callback or a symbolic selector evaluated at the
terminal sample. When omitted, sum the terminal quadrature states; at least one
quadrature must be declared. Constraints are `selector => (; t, bounds, side)`
pairs. Symbolic selectors are evaluated at saved samples; `t` defaults to the final
time, and `t = :all` selects all saved times. At reset times `side = :post` selects
the outgoing state; `:pre` and `:both` are explicit alternatives. Parameter values
remain right-continuous for all sides.
A trajectory callback without `t` may return a scalar or flat vector; with `t` it
must return one entry per trajectory sample, from which the requested entries are
selected. Bounds are `(lower, upper)`, with scalar or per-entry vector limits.

Supply `ps` and `st` together to reuse initialization, including parameters expanded
by [`add_parameter_segments!`](@ref). Otherwise call Lux setup once with `rng`.
Construction performs one forward solve to prepare the actual saved layout,
including `saveat`, and constraint dimensions. No initial trajectory is retained.
Missing requested samples are errors: the facade does not interpolate. Reconstruct
this model after changing schedules, saved times, or callback output dimensions.
State and parameter shooting equalities are appended automatically, with zero bounds.

Construct a SciMLBase `OptimizationProblem` using the reconstruction callable from [`pack`](@ref).
Load ComponentArrays for the default packing implementation. Block structure and
ModelingToolkit translation are separate features.

# Fields
$(FIELDS)
"""
@concrete terse struct CorleoneDynamicOptProblem
    "The shooting layer shared with forward predictions."
    layer
    "The scalar trajectory objective callback or prepared terminal selector."
    objective
    "Prepared path and terminal constraints in declaration order."
    constraints
    "Initialized numerical parameters, retained by reference."
    ps
    "Initialized Lux state, retained by reference."
    st
    "The actual saved time grid used to prepare constraint selection."
    times
    "Lower constraint bounds: user constraints followed by shooting equalities."
    lcons
    "Upper constraint bounds in the same order as `lcons`."
    ucons
    "The prepared number of state and parameter shooting equalities."
    nshooting::Int
end

"""
$(TYPEDEF)

Evaluate a prepared symbolic getter at the terminal saved state.

# Fields
$(FIELDS)
"""
struct TerminalObjective{G, S}
    "The SII getter acting on a single state/parameter/time snapshot."
    getter::G
    "The scalar reduction, or `identity` for a scalar selector."
    aggregate::S
end

function optimization_sample(traj, i)
    t = traj.t[i]
    p = SymbolicIndexingInterface.parameter_values_at_time(traj, traj, t)
    return ProblemState(; u = traj.u[i], p, t)
end
(f::TerminalObjective)(traj) = f.aggregate(f.getter(optimization_sample(traj, lastindex(traj.t))))

"""
$(TYPEDEF)

Select symbolic values from single saved state/parameter snapshots.

# Fields
$(FIELDS)
"""
struct SymbolicConstraintGetter{G}
    "The prepared SII getter."
    getter::G
end
"""
$(TYPEDEF)

Evaluate a user trajectory constraint callback.

# Fields
$(FIELDS)
"""
struct CallbackConstraintGetter{F}
    "The scalar, vector, or saved-series trajectory callback."
    f::F
end

constraint_vector(x::Real) = [x]
constraint_vector(x::AbstractVector{<:Real}) = x
constraint_vector(x) = throw(ArgumentError("Constraint callbacks/selectors must return a real scalar or a flat vector of real values, got $(typeof(x))."))

function constraint_values(g::SymbolicConstraintGetter, traj, indices)
    values = map(indices) do i
        constraint_vector(g.getter(optimization_sample(traj, i)))
    end
    return isempty(values) ? similar(first(traj.u), 0) : reduce(vcat, values)
end
constraint_values(g::CallbackConstraintGetter, traj, ::Nothing) = constraint_vector(g.f(traj))
function constraint_values(g::CallbackConstraintGetter, traj, indices)
    values = g.f(traj)
    values isa AbstractVector && length(values) == length(traj.t) ||
        throw(DimensionMismatch("A time-selected trajectory callback must return one entry per saved sample."))
    selected = map(i -> constraint_vector(values[i]), indices)
    return isempty(selected) ? similar(first(traj.u), 0) : reduce(vcat, selected)
end

"""
$(TYPEDEF)

A constraint prepared against a trajectory's actual saved layout.

# Fields
$(FIELDS)
"""
struct TrajectoryConstraint{G, I}
    "The prepared symbolic selector or trajectory callback."
    getter::G
    "Selected saved sample indices, or `nothing` for a whole-trajectory callback."
    indices::I
    "The fixed number of scalar constraint entries."
    n::Int
end
Base.length(c::TrajectoryConstraint) = c.n
function (c::TrajectoryConstraint)(traj)
    values = constraint_values(c.getter, traj, c.indices)
    length(values) == c.n || throw(DimensionMismatch("Constraint output length changed; reconstruct CorleoneDynamicOptProblem."))
    return values
end

is_symbolic_selector(x) = x isa Integer || !(symbolic_type(x) isa NotSymbolic)
function trajectory_objective(layer, loss)
    if isnothing(loss)
        symbols = Tuple(quadrature_symbols(layer.sys))
        isempty(symbols) && throw(ArgumentError("Supply a loss callback/symbol or declare quadrature states."))
        return TerminalObjective(getsym(layer.sys, symbols), sum)
    end
    return is_symbolic_selector(loss) ? TerminalObjective(getsym(layer.sys, loss), identity) : loss
end
scalar_objective(x::Real) = x
scalar_objective(x) = throw(ArgumentError("The objective must return a real scalar, got $(typeof(x))."))

function constraint_sample_indices(times, requested, side)
    side in (:pre, :post, :both) || throw(ArgumentError("Constraint side must be :pre, :post, or :both."))
    requested = requested === :all ? unique(times) : requested isa Real ? (requested,) : requested
    allunique(requested) || throw(ArgumentError("Constraint times must be unique."))
    return reduce(
        vcat, map(collect(requested)) do t
            matches = searchsortedfirst(times, t):searchsortedlast(times, t)
            isempty(matches) && throw(ArgumentError("Constraint time $t is not a saved sample. Add it to the problem's saveat or use a saved time; Trajectory does not interpolate."))
            side === :both ? collect(matches) : [side === :pre ? first(matches) : last(matches)]
        end; init = Int[]
    )
end
@non_differentiable constraint_sample_indices(::Any, ::Any, ::Any)

function constraint_bound(value, n)
    if value isa Real
        return fill(float(value), n)
    end
    value isa AbstractVector{<:Real} && length(value) == n ||
        throw(DimensionMismatch("Expected a scalar bound or a bound vector of length $n."))
    return float.(value)
end
function prepare_trajectory_constraint(layer, traj, definition::Pair)
    selector, spec = definition
    spec isa NamedTuple && haskey(spec, :bounds) || throw(ArgumentError("Each constraint needs a NamedTuple with bounds."))
    all(k -> k in (:t, :bounds, :side), keys(spec)) || throw(ArgumentError("Supported constraint options are t, bounds, and side."))
    bounds = spec.bounds
    bounds isa Union{Tuple, AbstractVector} && length(bounds) == 2 ||
        throw(ArgumentError("Constraint bounds must be (lower, upper)."))
    symbolic = is_symbolic_selector(selector)
    getter = symbolic ? SymbolicConstraintGetter(getsym(layer.sys, selector)) : CallbackConstraintGetter(selector)
    indices = if symbolic || haskey(spec, :t)
        requested = get(spec, :t, last(traj.t))
        constraint_sample_indices(traj.t, requested, get(spec, :side, :post))
    else
        haskey(spec, :side) && throw(ArgumentError("A callback side requires explicit t selection."))
        nothing
    end
    values = constraint_values(getter, traj, indices)
    n = length(values)
    lower, upper = constraint_bound(first(bounds), n), constraint_bound(last(bounds), n)
    all(lower .<= upper) || throw(ArgumentError("Constraint lower bounds must not exceed upper bounds or contain NaN."))
    return TrajectoryConstraint(getter, indices, n), lower, upper
end

function CorleoneDynamicOptProblem(
        layer::ShootingLayer, loss, constraints::Pair...;
        ps = nothing, st = nothing, rng::Random.AbstractRNG = Random.default_rng()
    )
    isnothing(ps) == isnothing(st) || throw(ArgumentError("Supply initialized ps and st together."))
    objective = trajectory_objective(layer, loss)
    if isnothing(ps)
        ps, st = LuxCore.setup(rng, layer)
    end
    trajectory, _ = layer(nothing, ps, st)
    scalar_objective(objective(trajectory))
    prepared = map(c -> prepare_trajectory_constraint(layer, trajectory, c), constraints)
    nshooting = get_number_of_shooting_constraints(layer, ps, st)
    lcons = vcat(map(x -> x[2], prepared)..., zeros(nshooting))
    ucons = vcat(map(x -> x[3], prepared)..., zeros(nshooting))
    return CorleoneDynamicOptProblem(
        layer, objective, map(first, prepared), ps, st, trajectory.t, lcons, ucons, nshooting
    )
end
CorleoneDynamicOptProblem(layer::ShootingLayer; loss = nothing, constraints = (), kwargs...) =
    CorleoneDynamicOptProblem(layer, loss, constraints...; kwargs...)

function check_optimization_layout(expected, actual, expected_count, actual_count)
    expected == actual && expected_count == actual_count ||
        throw(ArgumentError("Saved times or shooting schedules changed; reconstruct CorleoneDynamicOptProblem."))
    return nothing
end
@non_differentiable check_optimization_layout(::Any, ::Any, ::Any, ::Any)

function optimization_trajectory(prob, ps, st)
    traj, _ = prob.layer(nothing, ps, st)
    check_optimization_layout(
        prob.times, traj.t, prob.nshooting,
        get_number_of_shooting_constraints(prob.layer, ps, st)
    )
    return traj
end

"""
$(SIGNATURES)

Solve the prepared model once and evaluate its scalar trajectory objective.
Parameters and state default to those retained by [`CorleoneDynamicOptProblem`](@ref).
No numeric solution cache is reused across calls or differentiation contexts.
"""
function optimization_objective(prob::CorleoneDynamicOptProblem, ps = prob.ps, st = prob.st)
    return scalar_objective(prob.objective(optimization_trajectory(prob, ps, st)))
end

"""
$(SIGNATURES)

Solve once and return user constraints in declaration/sample order, followed by
state and parameter shooting equalities. This allocating form supports AD backends
that do not support array mutation. Dimensions and saved layout must match setup.
"""
function optimization_constraints(prob::CorleoneDynamicOptProblem, ps = prob.ps, st = prob.st)
    traj = optimization_trajectory(prob, ps, st)
    paths = map(c -> c(traj), prob.constraints)
    return vcat(paths..., shooting_constraints(traj))
end

"""
$(SIGNATURES)

Write [`optimization_constraints`](@ref) into `res` and return the buffer. Check its
length before solving/writing. Path callbacks are evaluated before writes; shooting
residuals are written directly with [`shooting_constraints!`](@ref). The buffer must
support the numerical scalar type, including Dual numbers when using ForwardDiff.
"""
function optimization_constraints!(res, prob::CorleoneDynamicOptProblem, ps = prob.ps, st = prob.st)
    length(res) == length(prob.lcons) || throw(DimensionMismatch("Expected $(length(prob.lcons)) constraint entries, got $(length(res))."))
    traj = optimization_trajectory(prob, ps, st)
    paths = map(c -> c(traj), prob.constraints)
    index = firstindex(res)
    for values in paths
        copyto!(res, index, values, firstindex(values), length(values))
        index += length(values)
    end
    shooting_constraints!(view(res, index:lastindex(res)), traj)
    return res
end
