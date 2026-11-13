"One experiment, or a tuple or NamedTuple of experiments."
const ObservationExperiments = Union{Experiment, Tuple{Vararg{Experiment}}, NamedTuple{<:Any, <:Tuple{Vararg{Experiment}}}}

as_experiments(experiment::Experiment) = (experiment,)
as_experiments(experiments::ObservationExperiments) = experiments

is_weighted(experiment::Experiment) = map(observation -> !isnothing(observation.σ), experiment.observations)

# The wrapper delegates Lux setup and bounds to the owner of the tunable tree.
mutable struct ExperimentPrediction{S, L, D, P, E} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    const shooting::S
    const layer::L
    const data::D
    const definition::P
    const evaluation::E
end

experiment_parameters(::Nothing, ps) = ps
experiment_parameters(::EvaluationLayer, ps) = ps.shooting
experiment_parameters(layer::ExperimentPrediction, ps) = experiment_parameters(layer.evaluation, ps)

function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::ExperimentPrediction)
    ps = LuxCore.initialparameters(rng, layer.layer)
    isnothing(layer.evaluation) && return ps
    return (; shooting = ps, evaluation = layer.evaluation.trainable)
end
LuxCore.parameterlength(layer::ExperimentPrediction) =
    LuxCore.parameterlength(layer.layer) + (isnothing(layer.evaluation) ? 0 : length(layer.evaluation.trainable))
LuxCore.statelength(layer::ExperimentPrediction) =
    LuxCore.statelength(layer.layer) + (isnothing(layer.evaluation) ? 0 : length(layer.evaluation.fixed))

for operation in (:get_lower_bound, :get_upper_bound, :get_integrality)
    @eval function Corleone.$operation(layer::ExperimentPrediction, ps, st = nothing)
        shooting = Corleone.$operation(layer.layer, experiment_parameters(layer, ps), st)
        isnothing(layer.evaluation) && return shooting
        evaluation = evaluation_parameter_metadata(Corleone.$operation, layer.evaluation, ps.evaluation)
        return evaluation_parameter_tree(ps, shooting, evaluation)
    end
end

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::ExperimentPrediction)
    return experiment_evaluation_state(layer.evaluation, LuxCore.initialstates(rng, layer.layer))
end

experiment_evaluation_state(::Nothing, st) = st
experiment_evaluation_state(layer::EvaluationLayer, st) =
    isempty(layer.fixed) ? st : merge(st, (; evaluation = layer.fixed))

log_prior(layer::ExperimentPrediction, ps, st) =
    log_prior(layer.layer, experiment_parameters(layer, ps), st) + evaluation_log_prior(layer.evaluation, ps)

function parameter_collections(layer::ExperimentPrediction, ps, st)
    shooting = parameter_collections(layer.layer, experiment_parameters(layer, ps), st)
    isnothing(layer.evaluation) && return shooting
    return Tuple(Iterators.flatten(evaluation_parameter_tree(ps, shooting, evaluation_collections(layer.evaluation, ps))))
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
    return experiment_prediction(layer.layer, layer, ps, st, grid)
end
function experiment_prediction(owner::Experiments, layer, ps, st, grid)
    state = merge(st, (; saveat = grid))
    # The tuple form keeps one concrete trajectory type per experiment; converting
    # the public Vector result with Tuple(...) loses its length during inference.
    if isnothing(layer.evaluation)
        trajectories, _ = Corleone.experiment_trajectories(layer.shooting, owner, ps, state)
    else
        trajectories, _ = Corleone.experiment_trajectories(layer.evaluation, owner, ps, (; shooting = state, evaluation = get(st, :evaluation, layer.evaluation.fixed)))
    end
    return NamedTuple{keys(layer.data)}(trajectories)
end
function experiment_prediction(owner::ShootingLayer, layer, ps, st, grid)
    state = merge(st, (; saveat = grid))
    parameters = experiment_parameters(layer, ps)
    return map(layer.data) do experiment
        first(layer.shooting(experiment, parameters, state))
    end
end

function get_number_of_shooting_constraints(layer::ExperimentPrediction, ps, st)
    n = get_number_of_shooting_constraints(layer.layer, experiment_parameters(layer, ps), st)
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
residual(observation::PreparedObservation, trajectory, evaluation = ()) =
    observation_residual(observation, trajectory, observation.indices, evaluation)

prepare_observable(selector, trajectory, eval_module, evaluator, layer) = selector
function prepare_observable(selector::Union{Symbol, Integer}, trajectory, eval_module, evaluator, layer)
    index = state_index(trajectory, selector)
    isnothing(index) && throw(ArgumentError("Unknown observed state $selector."))
    return index
end
prepare_observable(expression::Expr, trajectory, eval_module, evaluator, layer) =
    compile_observation_expression(expression, symbolic_container(trajectory), trajectory.t, layer, eval_module, evaluator)

function prepare_observations(data::Experiment, trajectory, eval_module, evaluator, layer)
    observations = map(data.observations) do observation
        selector = prepare_observable(observation.selector, trajectory, eval_module, evaluator, layer)
        indices = sample_indices(trajectory.t, observation.t)
        all(!isnothing, indices) || throw(ArgumentError("Every observation time must be saved in its experiment's trajectory."))
        PreparedObservation(selector, observation.t, observation.y, observation.σ, Int.(indices))
    end
    return Experiment(observations, data.u0, data.t)
end

mutable struct MeasurementObjective{O, D, R}
    const owner::O
    const data::D
    const reducer::R
end
# Experiments of one concrete type run in a loop over a single compiled method;
# mixed experiment types keep an unrolled map, so each result stays concrete.
# The explicit loop indexes the argument tuples instead of capturing them in a
# generator closure, which would copy every experiment by value.
function paired_map(f, a::NTuple{N, A}, b::NTuple{N, B}) where {N, A, B}
    value = f(a[1], b[1])
    out = Vector{typeof(value)}(undef, N)
    out[1] = value
    for i in 2:N
        out[i] = f(a[i], b[i])
    end
    return out
end
paired_map(f, a::Tuple, b::Tuple) = map(f, a, b)
paired_map(f, ::Tuple{}, ::Tuple{}) = ()

@noinline experiment_residuals(experiment, trajectory, evaluation) =
    reduce(vcat, map(observation -> residual(observation, trajectory, evaluation), experiment.observations))

# Named experiments are matched by name; positional experiments by position.
aligned_trajectories(data::NamedTuple, trajectories::NamedTuple) = values(NamedTuple{keys(data)}(trajectories))
aligned_trajectories(data, trajectories) = values(trajectories)

function measurement_residuals(data, trajectories, evaluation)
    aligned = aligned_trajectories(data, trajectories)
    groups = paired_map((experiment, trajectory) -> experiment_residuals(experiment, trajectory, evaluation), values(data), aligned)
    return reduce(vcat, groups)
end

reduce_measurements(reducer, data, trajectories, evaluation) = reducer(abs2.(measurement_residuals(data, trajectories, evaluation)))
reduce_measurements(::typeof(mean), data, trajectories, evaluation) = mean_squared_residual(data, trajectories, evaluation)

function evaluate_trajectory_objective(objective::MeasurementObjective, trajectories, ps, st)
    evaluation = evaluation_parameters(objective.owner.evaluation, ps, st)
    loss = reduce_measurements(objective.reducer, objective.data, trajectories, evaluation)
    return scalar_objective(loss) - log_prior(objective.owner, ps, st)
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
lcons, ucons, reducer = mean, observation_parameters = (;), eval_module, evaluator)` from an [`Experiments`](@ref) container or
observation [`Experiment`](@ref), tuple, or NamedTuple. Lux parameters and state
follow their owner, initialized once when omitted.

The reducer receives the squared, noise-scaled observation residuals in residual
order and returns a scalar. The default is `Statistics.mean`. Pass
`reducer = x -> sum(x) / 2` for the Gaussian half-sum convention.
Annotation priors add their negative log density once per owned value, outside
the reduction. Observable selectors accept
state symbols/indices, `(u, p, t) -> value` callbacks, or scalar Julia expressions
such as `:(p₁ * u₁)`. Expressions use current-time signal names and the independent
variable; each observation supplies its own sampling grid. They compile once
against the experiment's symbolic metadata. `eval_module` resolves custom helpers.

`observation_parameters = (w = Tunable(default = 0.5), offset = Constant(default = 0.0))`
declares shared observation-only values. Numeric declarations are shorthand for
unbounded tunables.
The fit creates an [`EvaluationLayer`](@ref) at `model.layer.evaluation`. The
prepared parameter tree stores the shooting values in `model.ps.shooting` and the
extra tunables in `model.ps.evaluation`. Constants are stored in
`model.st.evaluation`. If supplied, `ps` and `st` are the shooting or
experiments owner's Lux setup values.

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
        lcons = nothing, ucons = nothing, reducer = mean,
        eval_module::Module = Corleone, evaluator = default_evaluator(),
        observation_parameters::NamedTuple = (;)
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
    evaluation = isempty(observation_parameters) ? nothing : EvaluationLayer(shooting, nothing; observation_parameters...)
    layer = ExperimentPrediction(shooting, owner, data, (; constraints, lcons, ucons, eval_module, evaluator), evaluation)
    if !isnothing(evaluation)
        observation_symbol_indices(shooting.sys, evaluation)
        ps = (; shooting = ps, evaluation = evaluation.trainable)
        st = experiment_evaluation_state(evaluation, st)
    end
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
    prepared = map(data, trajectory) do experiment, prediction
        prepare_observations(experiment, prediction, eval_module, evaluator, evaluation)
    end
    objective = MeasurementObjective(layer, prepared, reducer)
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
    measurement_residuals(model.objective.data, optimization_trajectory(model, ps, st), evaluation_parameters(model.layer.evaluation, ps, st))
n_observations(model::ExperimentDynamicOptProblem) = n_observations(model.objective.data)

"""
$(SIGNATURES)

Return the mean squared observation residual across all experiments. Each
observed value counts once, with the same noise scaling as [`residual`](@ref).
Priors and shooting constraints are excluded. With no observed values, the
result is `NaN`.

Evaluate the prediction once and sum over observation groups without
concatenating their residual vectors. Parameters and state default to those
retained by the model.
"""
function mean_squared_residual(model::ExperimentDynamicOptProblem, ps = model.ps, st = model.st)
    return mean_squared_residual(model.objective.data, optimization_trajectory(model, ps, st), evaluation_parameters(model.layer.evaluation, ps, st))
end

function mean_squared_residual(data, trajectories, evaluation = ())
    totals = map(data, trajectories) do experiment, trajectory
        sum(map(observation -> sum(abs2, residual(observation, trajectory, evaluation)), experiment.observations))
    end
    return sum(values(totals)) / n_observations(data)
end
