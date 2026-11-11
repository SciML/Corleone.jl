# The wrapper delegates Lux setup and bounds to the owner of the tunable tree.
struct ExperimentPrediction{S, L, D} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    shooting::S
    layer::L
    data::D
end

function experiment_data(layer::Experiments)
    data = map(experiment -> experiment.observations, layer.experiments)
    all(experiment -> experiment isa Experiment, data) ||
        throw(ArgumentError("Every experiment must provide observations for estimation."))
    return data
end

experiment_metadata(layer::ExperimentPrediction) = experiment_metadata(layer.layer, layer.shooting, layer.data)
experiment_metadata(owner::Experiments, shooting, data) = owner.configurations
experiment_metadata(owner::ShootingLayer, shooting, data) = map(_ -> shooting, data)
experiment_grid(owner::Experiments, data, st) = get_saveat(owner, st)
experiment_grid(owner::ShootingLayer, data, st) = get_saveat(owner, data, st)

optimization_schedule(layer::ExperimentPrediction, st) = experiment_schedule(layer.layer, st)
# Reset flags in `active` are forward-pass scratch state, not part of the schedule.
experiment_schedule(owner::Experiments, st) =
    map(entry -> entry.timestops, experiment_fields(st.execution, owner.experiments))
experiment_schedule(layer::ShootingLayer, st) = optimization_schedule(layer, st)

# Optimization evaluations retain the supplied experiment conditions.
function (layer::ExperimentPrediction)(::Any, ps, st)
    grid = experiment_grid(layer.layer, layer.data, st)
    return optimization_prediction(layer, ps, st, grid), st
end

function optimization_prediction(layer::ExperimentPrediction, ps, st, grid)
    return experiment_prediction(layer.layer, layer.shooting, layer.data, ps, st, grid)
end
function experiment_prediction(owner::Experiments, shooting, data, ps, st, grid)
    trajectories, _ = shooting(owner, ps, merge(st, (; saveat = grid)))
    return NamedTuple{keys(data)}(Tuple(trajectories))
end
function experiment_prediction(owner::ShootingLayer, shooting, data, ps, st, grid)
    state = merge(st, (; saveat = grid))
    return map(data) do experiment
        first(shooting(experiment, ps, state))
    end
end

function get_number_of_shooting_constraints(layer::ExperimentPrediction, ps, st)
    n = get_number_of_shooting_constraints(layer.layer, ps, st)
    return layer.layer isa Experiments ? n : length(layer.data) * n
end
optimization_matching(::ExperimentPrediction, trajectories) = reduce(vcat, map(shooting_constraints, values(trajectories)))
optimization_matching!(res, layer::ExperimentPrediction, trajectories) = copyto!(res, optimization_matching(layer, trajectories))

# Retain observation data and compile only its scalar observable map. Sampling
# indices are resolved once against each experiment's actual saved trajectory.
struct PreparedObservation{F, T, Y, S, I}
    selector::F
    t::T
    y::Y
    σ::S
    indices::I
end
residual(observation::PreparedObservation, trajectory) = observation_residual(observation, trajectory, observation.indices)

struct ObservationExpression{F}
    f::F
end
(expression::ObservationExpression)(u, p, t) = expression.f((; states = u, parameters = p, time = (t,)))

prepare_observable(selector, metadata, trajectory, eval_module, evaluator) = selector
function prepare_observable(selector::Union{Symbol, Integer}, metadata, trajectory, eval_module, evaluator)
    index = state_index(trajectory, selector)
    isnothing(index) && throw(ArgumentError("Unknown observed state $selector."))
    return index
end
function prepare_observable(expression::Expr, metadata, trajectory, eval_module, evaluator)
    indices = expression_symbol_indices(metadata)
    for symbol in independent_variable_symbols(metadata.sys)
        name = expression_symbol_name(symbol)
        haskey(indices, name) && throw(ArgumentError("Independent variables and signals must have distinct names."))
        indices[name] = (:time, 1)
    end
    timepoints = Dict{Symbol, Vector{Real}}()
    collect_expression_times!(timepoints, expression, indices)
    all(isempty, values(timepoints)) ||
        throw(ArgumentError("Observation expressions use current-time signal names; sampling times belong in the observation grid."))
    input = gensym(:observation)
    body = replace_expression_samples(expression, indices, trajectory.t, input)
    return ObservationExpression(compile_evaluator(evaluator, Expr(:->, input, body), eval_module))
end
function prepare_observations(data::Experiment, metadata, trajectory, eval_module, evaluator)
    observations = map(data.observations) do observation
        selector = prepare_observable(observation.selector, metadata, trajectory, eval_module, evaluator)
        indices = sample_indices(trajectory.t, observation.t)
        all(!isnothing, indices) || throw(ArgumentError("Every observation time must be saved in its experiment's trajectory."))
        PreparedObservation(selector, observation.t, observation.y, observation.σ, Int.(indices))
    end
    return Experiment(observations, data.u0, data.t)
end

struct MeasurementObjective{O, D}
    owner::O
    data::D
end
(objective::MeasurementObjective)(trajectories) = sum(abs2, residual(objective.data, trajectories)) / 2
function evaluate_trajectory_objective(objective::MeasurementObjective, trajectories, ps, st)
    return objective(trajectories) - log_prior(objective.owner, ps, st)
end

# A key in the type keeps heterogeneous experiment selection inferable.
struct ExperimentConstraint{K, F, I}
    f::F
end
(constraint::ExperimentConstraint{K, F, false})(trajectories) where {K, F} = constraint.f(getfield(trajectories, K))
(constraint::ExperimentConstraint{K, F, true})(res, trajectories) where {K, F} = constraint.f(res, getfield(trajectories, K))

function experiment_constraint(constraint, data)
    if constraint isa Pair
        key, expression = constraint
        key in keys(data) || throw(ArgumentError("Unknown experiment $key in constraint."))
        return key, expression
    elseif constraint isa Expr
        length(data) == 1 || throw(ArgumentError("Tag symbolic constraints with experiment => expression when fitting several experiments."))
        return only(keys(data)), constraint
    end
    return nothing
end

"""
$(SIGNATURES)

Build `CorleoneDynamicOptProblem(shooting, experiments, constraints...; ps, st,
lcons, ucons, eval_module, evaluator)` from an [`Experiments`](@ref) container or
observation [`Experiment`](@ref), tuple, or NamedTuple. Lux parameters and state
follow their owner, initialized once when omitted.

Observations define half the weighted residual sum of squares. Annotation priors
add their negative log density once per owned value. Observable selectors accept
state symbols/indices, `(u, p, t) -> value` callbacks, or scalar Julia expressions
such as `:(p₁ * u₁)`. Expressions use current-time signal names and the independent
variable; each observation supplies its own sampling grid. They compile once
against the experiment's symbolic metadata. `eval_module` resolves custom helpers.

A tagged constraint `:experiment => :(u₁(2.0))` evaluates on that experiment.
Untagged expressions apply to a single experiment. Callbacks receive the complete
NamedTuple or tuple of trajectories; tagged callbacks receive one trajectory.
Constraint bounds and automatic shooting equalities follow the ordinary dynamic
optimization interface. Reconstruct the model after changing observation times,
selectors, schedules, or saved layouts. Observation values and noise arrays remain
referenced by the prepared model.
"""
function CorleoneDynamicOptProblem(
        shooting::ShootingLayer, experiments::Union{Experiments, ObservationExperiments}, constraints...;
        ps = nothing, st = nothing, rng::Random.AbstractRNG = Random.default_rng(),
        lcons = nothing, ucons = nothing, eval_module::Module = @__MODULE__, evaluator = default_evaluator()
    )
    owner = experiments isa Experiments ? experiments : shooting
    data = experiments isa Experiments ? experiment_data(experiments) : as_experiments(experiments)
    isempty(data) && throw(ArgumentError("Supply at least one experiment."))
    weighted = reduce(vcat, map(collect ∘ is_weighted, values(data)))
    all(weighted) || !any(weighted) || throw(ArgumentError("Either all observations or none must specify a noise scale σ."))
    isnothing(ps) == isnothing(st) || throw(ArgumentError("Supply initialized ps and st together."))
    isnothing(ps) && ((ps, st) = LuxCore.setup(rng, owner))
    if owner isa ShootingLayer && length(data) > 1 && get_number_of_shooting_constraints(owner, ps, st) > 0
        throw(ArgumentError("Shared shooting-node values require a single observation experiment; use Experiments for local nodes."))
    end
    bounds = prepare_constraint_bounds(lcons, ucons, length(constraints))
    layer = ExperimentPrediction(shooting, owner, data)
    metadata = experiment_metadata(layer)
    indices = map(expression_symbol_indices, metadata)
    timepoints = map(_ -> Dict{Symbol, Vector{Real}}(), data)
    descriptions = map(constraint -> experiment_constraint(constraint, data), constraints)
    for description in descriptions
        isnothing(description) && continue
        key, expression = description
        expression isa Expr && collect_expression_times!(timepoints[key], expression, indices[key])
    end
    grid = experiment_grid(owner, data, st)
    expression_times = Iterators.flatten(map(values, values(timepoints)))
    grid = expression_sample_grid(shooting.problem.tspan, grid, expression_times)
    trajectory = optimization_prediction(layer, ps, st, grid)
    prepared = map((experiment, symbols, prediction) -> prepare_observations(experiment, symbols, prediction, eval_module, evaluator), data, metadata, trajectory)
    objective = MeasurementObjective(owner, prepared)
    prepared_constraints = map(constraints, descriptions) do constraint, description
        isnothing(description) && return constraint
        key, expression = description
        f = expression isa Expr ? compile_trajectory_expression(expression, indices[key], optimization_times(trajectory[key]), eval_module, evaluator) : expression
        return ExperimentConstraint{key, typeof(f), !applicable(f, trajectory[key])}(f)
    end
    return prepare_dynamic_opt_problem(layer, objective, prepared_constraints, bounds, ps, st, timepoints, grid, trajectory, nothing, eval_module, evaluator)
end

const ExperimentDynamicOptProblem = CorleoneDynamicOptProblem{<:ExperimentPrediction, <:MeasurementObjective}

"""
$(SIGNATURES)

Evaluate the prepared observation residual vector of an experiment optimization
model. Parameters and state default to those retained by the model. Priors are
handled separately by optimization and least-squares problem construction.
"""
residual(model::ExperimentDynamicOptProblem, ps = model.ps, st = model.st) =
    residual(model.objective.data, optimization_trajectory(model, ps, st))
n_observations(model::ExperimentDynamicOptProblem) = n_observations(model.objective.data)
