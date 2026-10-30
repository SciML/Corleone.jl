"""
$(TYPEDEF)

Prepare `CorleoneDynamicOptProblem(layer, objective, constraints...; lcons, ucons,
ps, st, rng, eval_module)`. The objective is a scalar trajectory callback `f(prediction)` or
`f(prediction, st)`, a Julia
`Expr`, or a terminal state/parameter selector. Omitting it sums terminal quadratures.

Callback-based Lux layers may return a collection of trajectories. Timed
expressions and terminal selectors require a `ShootingLayer`.

Each constraint is a scalar/vector callback `f(traj)`, an in-place callback
`f!(res, traj)` whose return is ignored, or a numerical Julia expression. Supply
`lcons` and `ucons` with matching structure and one top-level entry per constraint:
`[0.0, [0.0, 0.0], 0.0]` declares output sizes `(1, 2, 1)`. With a single constraint,
a flat bound vector is also accepted. Bounds are flattened in declaration order;
state and parameter shooting equalities are appended automatically with zero bounds.

Timed signal calls such as `:(u₁(10.0) + p₁(5.0))` collect their timepoints during
setup. Prediction uses the union of these points and `st.timestops`. Expressions
are compiled against the actual saved grid, including internal `saveat` samples.
A timed reset query selects the post-reset state; parameters are right-continuous.
Bare signal names within expressions denote their full series on that saved grid.
Constraint comparisons are not parsed: use numerical expressions and lcons/ucons.

`eval_module` resolves ordinary expression functions and constants (default:
Corleone, including Base functions). Pass your own module for custom helpers.
Signal times must be real literals within the time span; interpolate computed times.
SymbolicUtils parsing is deferred. `evaluator = default_evaluator()` selects the
loaded RuntimeGeneratedFunctions extension, with eval as the built-in fallback.

Supply `ps`/`st` together to retain them by reference, including expanded parameter
intervals. Otherwise setup runs once. One initial prediction validates outputs and
prepares indices; its numerical solution is not retained. Reprepare after changing
schedules, saved layout, or output dimensions. Load ComponentArrays for the default
SciMLBase optimization bridge, or supply another packing implementation.

# Fields
$(FIELDS)
"""
struct CorleoneDynamicOptProblem{L, O, C, P, S, T, G, B, V, LB, UB}
    "The Lux layer shared with forward predictions."
    layer::L
    "The scalar objective callback or prepared expression/terminal selector."
    objective::O
    "Prepared constraint evaluators in declaration order."
    constraints::C
    "Initialized numerical parameters, retained by reference."
    ps::P
    "Initialized Lux state, retained by reference."
    st::S
    "Signal names mapped to requested expression times; discrete setup metadata."
    timepoints::T
    "The evaluation grid, or nothing when prediction uses the layer's own grids."
    evaluation_grid::G
    "The original cached shooting-stop tree used to detect schedule changes."
    source_timestops::B
    "The actual saved prediction grid or grid tree, retained for layout checks."
    times::V
    "Flattened user lower bounds followed by shooting equality bounds."
    lcons::LB
    "Flattened user upper bounds followed by shooting equality bounds."
    ucons::UB
    "The prepared number of state and parameter shooting equalities."
    nshooting::Int

    function CorleoneDynamicOptProblem(
            ::Val{:Prepared}, layer::L, objective::O, constraints::C, ps::P, st::S,
            timepoints::T, evaluation_grid::G, source_timestops::B, times::V,
            lcons::LB, ucons::UB, nshooting::Int
        ) where {L, O, C, P, S, T, G, B, V, LB, UB}
        return new{L, O, C, P, S, T, G, B, V, LB, UB}(
            layer, objective, constraints, ps, st, timepoints, evaluation_grid,
            source_timestops, times, lcons, ucons, nshooting
        )
    end
end

function terminal_optimization_sample(traj)
    t = last(current_time(traj))
    p = SymbolicIndexingInterface.parameter_values_at_time(traj, traj, t)
    return ProblemState(; u = last(state_values(traj)), p, t)
end

function trajectory_objective(layer, loss)
    if isnothing(loss) || loss isa Union{Symbol, Integer}
        throw(ArgumentError("Default objectives and terminal selectors require a ShootingLayer; supply a callback."))
    end
    return loss
end
scalar_objective(x::Bool) = throw(ArgumentError("An objective must return a numerical scalar, not a comparison."))
scalar_objective(x::Real) = x
scalar_objective(x) = throw(ArgumentError("The objective must return a real scalar, got $(typeof(x))."))

constraint_vector(x::Bool) = throw(ArgumentError("Constraints must return numerical values; use lcons/ucons for bounds."))
constraint_vector(x::AbstractVector{Bool}) = throw(ArgumentError("Constraints must return numerical values; use lcons/ucons for bounds."))
constraint_vector(x::Real) = [x]
constraint_vector(x::AbstractVector{<:Real}) = x
constraint_vector(x) = throw(ArgumentError("Constraints must return a real scalar or flat vector, got $(typeof(x))."))

"""
$(TYPEDEF)

A fixed-size constraint callback with a setup-time choice of calling convention.

# Fields
$(FIELDS)
"""
struct PreparedTrajectoryConstraint{F, I}
    "The trajectory callback or compiled numerical expression."
    f::F
    "The output length defined by this constraint's bound subtree."
    n::Int
end
Base.length(c::PreparedTrajectoryConstraint) = c.n

function constraint_eltype(traj)
    T = eltype(first(state_values(traj)))
    P = eltype(parameter_values(traj))
    return P <: Real ? promote_type(T, P) : T
end
constraint_buffer(traj, n) = similar(first(state_values(traj)), constraint_eltype(traj), n)
function constraint_buffer(trajectories::NamedTuple, n)
    T = promote_type(map(constraint_eltype, values(trajectories))...)
    return similar(first(state_values(first(values(trajectories)))), T, n)
end
function (c::PreparedTrajectoryConstraint{F, false})(traj) where {F}
    values = constraint_vector(c.f(traj))
    length(values) == c.n || throw(DimensionMismatch("Constraint output length changed; reconstruct CorleoneDynamicOptProblem."))
    return values
end
function (c::PreparedTrajectoryConstraint{F, true})(traj) where {F}
    res = constraint_buffer(traj, c.n)
    c.f(res, traj)
    return res
end
function (c::PreparedTrajectoryConstraint{F, false})(res, traj) where {F}
    copyto!(res, c(traj))
    return res
end
function (c::PreparedTrajectoryConstraint{F, true})(res, traj) where {F}
    c.f(res, traj)
    return res
end

function prepare_trajectory_constraint(f, traj, n)
    if applicable(f, traj)
        c = PreparedTrajectoryConstraint{typeof(f), false}(f, n)
        c(traj) # Validate the fixed output length.
        return c
    end
    buffer = constraint_buffer(traj, n)
    applicable(f, buffer, traj) || throw(ArgumentError("A constraint must support f(traj) or f!(res, traj)."))
    f(buffer, traj) # Return value is deliberately ignored.
    return PreparedTrajectoryConstraint{typeof(f), true}(f, n)
end

bound_shape(x::Real) = nothing
bound_shape(x::Union{Tuple, AbstractVector}) = map(bound_shape, Tuple(x))
bound_shape(x) = throw(ArgumentError("Constraint bounds must contain real scalars or collections of real scalars."))
flatten_constraint_bound(x::Real) = [float(x)]
flatten_constraint_bound(x::Union{Tuple, AbstractVector}) =
    isempty(x) ? Float64[] : reduce(vcat, map(flatten_constraint_bound, x))

function constraint_bound_entries(bounds, n)
    bounds isa Union{Real, Tuple, AbstractVector} || throw(ArgumentError("Constraint bounds must be real scalars or collections."))
    if n == 1
        bounds isa Real && return (bounds,)
        # A singleton nested collection explicitly wraps the only constraint.
        length(bounds) == 1 && first(bounds) isa Union{Tuple, AbstractVector} && return (first(bounds),)
        return (bounds,)
    end
    bounds isa Union{Tuple, AbstractVector} && length(bounds) == n ||
        throw(DimensionMismatch("Supply one top-level lcons/ucons entry per constraint ($n entries)."))
    return Tuple(bounds)
end
function prepare_constraint_bounds(lcons, ucons, n)
    if n == 0
        all(x -> isnothing(x) || isempty(x), (lcons, ucons)) || throw(DimensionMismatch("No user constraints were supplied."))
        return ()
    end
    (isnothing(lcons) || isnothing(ucons)) &&
        throw(ArgumentError("Supply lcons and ucons with one matching bound subtree per constraint."))
    lower, upper = constraint_bound_entries(lcons, n), constraint_bound_entries(ucons, n)
    return map(lower, upper) do lo, hi
        bound_shape(lo) == bound_shape(hi) || throw(DimensionMismatch("Each lcons subtree must match its ucons subtree."))
        lb, ub = flatten_constraint_bound(lo), flatten_constraint_bound(hi)
        all(lb .<= ub) || throw(ArgumentError("Constraint lower bounds must not exceed upper bounds or contain NaN."))
        (lb, ub)
    end
end

function CorleoneDynamicOptProblem(
        layer::LuxCore.AbstractLuxLayer, loss, constraints...;
        ps = nothing, st = nothing, rng::Random.AbstractRNG = Random.default_rng(),
        lcons = nothing, ucons = nothing, eval_module::Module = @__MODULE__,
        evaluator = default_evaluator()
    )
    isnothing(ps) == isnothing(st) || throw(ArgumentError("Supply initialized ps and st together."))
    bounds = prepare_constraint_bounds(lcons, ucons, length(constraints))
    uses_expressions = loss isa Expr || any(c -> c isa Expr, constraints)
    uses_expressions && !(layer isa ShootingLayer) && throw(ArgumentError("Timed expressions require a ShootingLayer; use callbacks for other layers."))
    indices = uses_expressions ? expression_symbol_indices(layer) : Dict{Symbol, Int}()
    timepoints = Dict{Symbol, Vector{Real}}()
    loss isa Expr && collect_expression_times!(timepoints, loss, indices)
    for c in constraints
        c isa Expr && collect_expression_times!(timepoints, c, indices)
    end
    foreach(times -> sort!(unique!(times)), values(timepoints))
    if isnothing(ps)
        ps, st = LuxCore.setup(rng, layer)
    end
    grid = optimization_grid(layer, st, timepoints)
    trajectory = optimization_prediction(layer, ps, st, grid)
    objective = loss isa Expr ? compile_trajectory_expression(loss, indices, optimization_times(trajectory), eval_module, evaluator) : trajectory_objective(layer, loss)
    scalar_objective(evaluate_trajectory_objective(objective, trajectory, st))
    prepared = map(constraints, bounds) do c, bound
        f = c isa Expr ? compile_trajectory_expression(c, indices, optimization_times(trajectory), eval_module, evaluator) : trajectory_objective(layer, c)
        prepare_trajectory_constraint(f, trajectory, length(first(bound)))
    end
    nshooting = get_number_of_shooting_constraints(layer, ps, optimization_state(layer, st, grid))
    lower = vcat(map(first, bounds)..., zeros(nshooting))
    upper = vcat(map(last, bounds)..., zeros(nshooting))
    return CorleoneDynamicOptProblem(
        Val(:Prepared), layer, objective, prepared, ps, st, timepoints, grid, deepcopy(optimization_schedule(layer, st)),
        optimization_times(trajectory), lower, upper, nshooting
    )
end
CorleoneDynamicOptProblem(layer::LuxCore.AbstractLuxLayer; loss = nothing, constraints = (), kwargs...) =
    CorleoneDynamicOptProblem(layer, loss, constraints...; kwargs...)

function check_optimization_layout(expected, actual, expected_count, actual_count)
    expected == actual && expected_count == actual_count ||
        throw(ArgumentError("Saved times or shooting schedules changed; reconstruct CorleoneDynamicOptProblem."))
    return nothing
end
@non_differentiable check_optimization_layout(::Any, ::Any, ::Any, ::Any)
function check_optimization_schedule(expected, actual)
    expected == actual || throw(ArgumentError("Shooting schedules changed; reconstruct CorleoneDynamicOptProblem."))
    return nothing
end
@non_differentiable check_optimization_schedule(::Any, ::Any)

function optimization_trajectory(prob, ps, st)
    check_optimization_schedule(prob.source_timestops, optimization_schedule(prob.layer, st))
    traj = optimization_prediction(prob.layer, ps, st, prob.evaluation_grid)
    check_optimization_layout(prob.times, optimization_times(traj), prob.nshooting, get_number_of_shooting_constraints(prob.layer, ps, st))
    return traj
end

"""
$(SIGNATURES)

Solve once on the prepared evaluation grid and evaluate the scalar objective.
Parameters/state default to those retained by the model. No numerical solution
cache is shared between calls or AD contexts.
"""
optimization_objective(prob::CorleoneDynamicOptProblem, ps = prob.ps, st = prob.st) =
    scalar_objective(evaluate_trajectory_objective(prob.objective, optimization_trajectory(prob, ps, st), st))

"""
$(SIGNATURES)

Solve once and concatenate constraint outputs, followed by shooting equalities.
Out-of-place callbacks and compiled expressions support mutation-free reverse AD.
An in-place callback is evaluated into a fresh, appropriately typed buffer; its
AD support depends on the callback and backend's support for mutation.
"""
function optimization_constraints(prob::CorleoneDynamicOptProblem, ps = prob.ps, st = prob.st)
    traj = optimization_trajectory(prob, ps, st)
    return vcat(map(c -> c(traj), prob.constraints)..., optimization_matching(prob.layer, traj))
end

"""
$(SIGNATURES)

Write constraints into res in declaration order, followed by shooting equalities.
Check total length before solving/writing. In-place user callbacks receive their
own output slice directly; their return values are ignored. The buffer must support
the current numerical scalar type, including AD values.
"""
function optimization_constraints!(res, prob::CorleoneDynamicOptProblem, ps = prob.ps, st = prob.st)
    length(res) == length(prob.lcons) || throw(DimensionMismatch("Expected $(length(prob.lcons)) constraint entries, got $(length(res))."))
    traj = optimization_trajectory(prob, ps, st)
    index = firstindex(res)
    for c in prob.constraints
        c(view(res, index:(index + length(c) - 1)), traj)
        index += length(c)
    end
    optimization_matching!(view(res, index:lastindex(res)), prob.layer, traj)
    return res
end

# Callback-based layers can specialize these hooks without duplicating optimizer
# construction. Shooting-specific grid handling lives alongside ShootingLayer.
optimization_grid(layer, st, timepoints) = nothing
optimization_schedule(layer, st) = nothing
optimization_state(layer, st, grid) = st
optimization_prediction(layer, ps, st, grid) = first(layer(nothing, ps, st))
optimization_times(trajectory) = current_time(trajectory)
optimization_times(trajectories::NamedTuple) = map(optimization_times, trajectories)
optimization_matching(layer, trajectory) = shooting_constraints(trajectory)
optimization_matching!(res, layer, trajectory) = shooting_constraints!(res, trajectory)

# The optional state argument supports objectives such as information criteria
# with a prior supplied by OptimizationProblem.p rather than captured at setup.
evaluate_trajectory_objective(f, trajectory, st) = applicable(f, trajectory) ? f(trajectory) : f(trajectory, st)
@non_differentiable optimization_schedule(::Any, ::Any)
@non_differentiable optimization_times(::Any)
