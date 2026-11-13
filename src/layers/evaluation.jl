struct EvaluationSparsity
    jacobian::SparseMatrixCSC{Bool, Int}
    hessian::SparseMatrixCSC{Bool, Int}
end

"""
    EvaluationLayer(shooting::ShootingLayer, objective, constraints...; parameters...)

Wrap a shooting layer with additional scalar parameters for
[`CorleoneDynamicOptProblem`](@ref) expressions. Pass the objective first,
then the constraints in declaration order. For example, prepare
`EvaluationLayer(shooting, :(gain * u₁(1.0)), :gain; gain = 1.0)` with
`CorleoneDynamicOptProblem(layer; lcons = 0.0, ucons = Inf)`.

Use `gain = Tunable(default = 1.0, lowerbound = 0.0)` or
`offset = Constant(default = 0.0)` to declare a parameter. The keyword provides
the name, which must be distinct from shooting signals. A numeric keyword is
shorthand for an unbounded tunable. Tunables are stored in `ps.evaluation` in
internal coordinates and constants in `st.evaluation` in natural coordinates.
The shooting layer receives `ps.shooting`.

Call `layer(experiments, ps, st)` to run an [`Experiments`](@ref) batch.
Set `ps.shooting` and `st.shooting` to the results of Lux setup on
`experiments`. The call returns the batch trajectories and the updated
wrapper state. The evaluation parameter and state subtrees stay separate from shooting.

The prepared problem's `layer.sparsity` stores the packed constraint Jacobian
and Lagrangian Hessian patterns as `jacobian` and `hessian`. For ODE shooting,
preparation fills this snapshot when the default vectorizer is loaded. Otherwise
the first packed sparsity query fills it. The snapshot uses the packing at
preparation time.
"""
mutable struct EvaluationLayer{S <: ShootingLayer, O, C <: Tuple, P <: NamedTuple, T <: NamedTuple, F <: NamedTuple} <: LuxCore.AbstractLuxLayer
    const shooting::S
    const objective::O
    const constraints::C
    const evaluation::P
    const trainable::T
    const fixed::F
    sparsity::Union{Nothing, EvaluationSparsity}

    function EvaluationLayer(shooting::ShootingLayer, objective, constraints...; parameters...)
        definitions = (; parameters...)
        isempty(definitions) && throw(ArgumentError("Supply at least one evaluation parameter."))
        signals = expression_symbol_indices(shooting)
        names = keys(definitions)
        evaluation = NamedTuple{names}(
            map(names) do name
                haskey(signals, name) && throw(ArgumentError("Evaluation parameter $name duplicates a shooting signal."))
                evaluation_annotation(name, definitions[name])
            end
        )
        trainable_names = filter(name -> is_tunable(evaluation[name]), names)
        fixed_names = filter(name -> !is_tunable(evaluation[name]), names)
        trainable = NamedTuple{trainable_names}(
            map(trainable_names) do name
                annotation = evaluation[name]
                internal_value(annotation.transform, annotation.default)
            end
        )
        fixed = NamedTuple{fixed_names}(map(name -> evaluation[name].default, fixed_names))
        return new{typeof(shooting), typeof(objective), typeof(constraints), typeof(evaluation), typeof(trainable), typeof(fixed)}(
            shooting, objective, constraints, evaluation, trainable, fixed, nothing
        )
    end

    # Each prepared problem owns its packed patterns.
    EvaluationLayer(layer::EvaluationLayer{S, O, C, P, T, F}) where {S, O, C, P, T, F} =
        new{S, O, C, P, T, F}(layer.shooting, layer.objective, layer.constraints, layer.evaluation, layer.trainable, layer.fixed, nothing)
end

evaluation_annotation(name, value::Real) = evaluation_annotation(name, Tunable(; default = value))
evaluation_annotation(name, value) = throw(ArgumentError("Evaluation parameter $name must be a real scalar, Tunable, or Constant."))
function evaluation_annotation(name, annotation::DynamicVariable)
    (has_index(annotation) || has_sym(annotation)) &&
        throw(ArgumentError("Declare evaluation parameter $name without an annotation selector. The keyword supplies the name."))
    default = annotation.default
    default isa Real && !(default isa Bool) && isfinite(default) ||
        throw(ArgumentError("Evaluation parameter $name requires a finite real scalar default."))
    transform = resolve_transform(annotation.transform, (-Inf, Inf))
    validate_transform(transform)
    # The optimizer's integer mask applies to internal coordinates.
    is_integer_variable(annotation) && !(transform isa BoundedTransform) &&
        throw(ArgumentError("Integer evaluation parameter $name requires the identity (bounded) transform; the optimizer sees transformed values."))
    lower, upper = get_bounds(transform)
    lower <= upper || throw(ArgumentError("Lower bound exceeds upper bound for evaluation parameter $name."))
    validate_annotation_default(default, transform)
    validate_prior(annotation.prior)
    return remake(annotation; default = float(default), transform)
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::EvaluationLayer)
    return (; shooting = LuxCore.initialparameters(rng, layer.shooting), evaluation = layer.trainable)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::EvaluationLayer)
    return (; shooting = LuxCore.initialstates(rng, layer.shooting), evaluation = layer.fixed)
end

LuxCore.parameterlength(layer::EvaluationLayer) = LuxCore.parameterlength(layer.shooting) + length(layer.trainable)
LuxCore.statelength(layer::EvaluationLayer) = LuxCore.statelength(layer.shooting) + length(layer.fixed)

function (layer::EvaluationLayer)(x, ps, st)
    traj, shooting = LuxCore.apply(layer.shooting, x, ps.shooting, st.shooting)
    return traj, (; shooting, evaluation = st.evaluation)
end

function (layer::EvaluationLayer)(experiments::Experiments, ps, st)
    trajectories, shooting = layer.shooting(experiments, ps.shooting, st.shooting)
    return trajectories, (; shooting, evaluation = st.evaluation)
end

function experiment_trajectories(layer::EvaluationLayer, experiments::Experiments, ps, st)
    trajectories, shooting = experiment_trajectories(layer.shooting, experiments, ps.shooting, st.shooting)
    return trajectories, (; shooting, evaluation = st.evaluation)
end

for operation in (:get_lower_bound, :get_upper_bound, :get_integrality)
    @eval function $operation(layer::EvaluationLayer, ps, st = nothing)
        shooting = $operation(layer.shooting, ps.shooting, bound_child_state(st, :shooting))
        return compose_evaluation($operation, layer, ps, shooting)
    end
end

"""
    compose_evaluation(operation, evaluation, ps, inner)

Combine `inner`, the result of `operation` on the layer wrapped by `evaluation`,
with the evaluation parameters of `evaluation::EvaluationLayer`. `ps` is the
wrapper tree with fields `shooting` and `evaluation`; `inner` is computed from
its `shooting` subtree. Pass `evaluation = nothing` to return `inner` unchanged.

| `operation` | `inner` | Result |
|:--|:--|:--|
| `get_lower_bound`, `get_upper_bound`, `get_integrality` | metadata tree | `(; shooting = inner, evaluation)` restricted to the fields of `ps` |
| `parameter_collections` | collection tuple | `inner` followed by one collection per tunable |
| `log_prior` | scalar | `inner` plus the evaluation parameters' log prior |

Sublibraries use this to support evaluation parameters in layers that wrap
something other than a single [`ShootingLayer`](@ref).
"""
compose_evaluation(operation, ::Nothing, ps, inner) = inner
for operation in (:get_lower_bound, :get_upper_bound, :get_integrality)
    @eval compose_evaluation(::typeof($operation), layer::EvaluationLayer, ps, inner) =
        evaluation_parameter_tree(ps, inner, evaluation_parameter_metadata($operation, layer, ps.evaluation))
end
compose_evaluation(::typeof(parameter_collections), layer::EvaluationLayer, ps, inner) =
    Tuple(Iterators.flatten(evaluation_parameter_tree(ps, inner, evaluation_collections(layer, ps))))
compose_evaluation(::typeof(log_prior), layer::EvaluationLayer, ps, inner) = inner + evaluation_log_prior(layer, ps)

function evaluation_parameter_tree(ps, shooting, evaluation)
    parts = (; shooting, evaluation)
    names = optimization_parameter_names(ps, keys(parts))
    return NamedTuple{names}(map(name -> getproperty(parts, name), names))
end

function expression_symbol_indices(layer::EvaluationLayer)
    return add_evaluation_symbols!(expression_symbol_indices(layer.shooting), layer)
end

add_evaluation_symbols!(indices, ::Nothing) = indices
function add_evaluation_symbols!(indices, layer::EvaluationLayer)
    for (i, name) in enumerate(keys(layer.evaluation))
        haskey(indices, name) && throw(ArgumentError("Evaluation parameter $name duplicates a shooting signal."))
        indices[name] = (:evaluation, i)
    end
    return indices
end

function evaluation_parameter_metadata(operation, layer, values)
    names = optimization_parameter_names(values, keys(layer.trainable))
    return NamedTuple{names}(map(name -> evaluation_metadata(operation, layer.evaluation[name]), names))
end

evaluation_metadata(::typeof(get_lower_bound), annotation) = first(get_bounds(annotation.transform))
evaluation_metadata(::typeof(get_upper_bound), annotation) = last(get_bounds(annotation.transform))
evaluation_metadata(::typeof(get_integrality), annotation) = is_integer_variable(annotation)

"""
    evaluation_parameters(layer, ps, st)

Return the natural values of the evaluation parameters of `layer::EvaluationLayer`
in declaration order, reading tunables from `ps.evaluation` and constants from
`st.evaluation` (or the layer defaults). Other layers, including `nothing`, return `()`.
Pass the result as the `evaluation` argument of compiled trajectory and observation
expressions.
"""
evaluation_parameters(layer, ps, st) = ()
evaluation_parameters(layer::EvaluationLayer, ps, st) =
    evaluation_values(layer, ps.evaluation, get(st, :evaluation, layer.fixed))

function evaluation_values(layer::EvaluationLayer, values, fixed)
    return map(keys(layer.evaluation)) do name
        annotation = layer.evaluation[name]
        if is_tunable(annotation)
            hasproperty(values, name) || throw(ArgumentError("Missing evaluation parameter $name."))
            return natural_value(annotation.transform, getproperty(values, name))
        end
        return getproperty(fixed, name)
    end
end

evaluation_coordinate_ids(layer, ps) = ()
evaluation_coordinate_ids(layer::EvaluationLayer, ps) = map(keys(layer.evaluation)) do name
    is_tunable(layer.evaluation[name]) ? getproperty(ps.evaluation, name) : nothing
end

function evaluation_collections(layer::EvaluationLayer, ps)
    names = optimization_parameter_names(ps.evaluation, keys(layer.trainable))
    return map(name -> (layer.evaluation[name], (; ps = [getproperty(ps.evaluation, name)]), ()), names)
end

evaluation_log_prior(layer, ps) = 0.0
function evaluation_log_prior(layer::EvaluationLayer, ps)
    # Splatting into `+` keeps the reduction differentiable under Zygote, which
    # has no rule for `sum(f, itr; init)`.
    terms = map(keys(layer.trainable)) do name
        annotation = layer.evaluation[name]
        isnothing(annotation.prior) && return 0.0
        x = getproperty(ps.evaluation, name)
        return log_density(annotation.prior, natural_value(annotation.transform, x)) + log_jacobian(annotation.transform, x)
    end
    return +(0.0, terms...)
end

with_evaluation_prior(layer, ps, objective) = objective
function with_evaluation_prior(layer::EvaluationLayer, ps, objective)
    any(name -> !isnothing(layer.evaluation[name].prior), keys(layer.trainable)) || return objective
    return objective - evaluation_log_prior(layer, ps)
end

log_prior(layer::EvaluationLayer, ps, st) =
    compose_evaluation(log_prior, layer, ps, log_prior(layer.shooting, ps.shooting, st.shooting))

parameter_collections(layer::EvaluationLayer, ps, st, inserted) =
    compose_evaluation(parameter_collections, layer, ps, parameter_collections(layer.shooting, ps.shooting, st.shooting, inserted))

function evaluation_prior_columns(layer::EvaluationLayer, ids)
    return [BitSet((getproperty(ids.evaluation, name),)) for name in keys(layer.trainable) if !isnothing(layer.evaluation[name].prior)]
end
evaluation_prior_columns(layer, ids) = BitSet[]

is_evaluation_parameter(layer, name) = layer isa EvaluationLayer && name isa Symbol && haskey(layer.evaluation, name)

optimization_grid(layer::EvaluationLayer, st, timepoints) = optimization_grid(layer.shooting, st.shooting, timepoints)
optimization_schedule(layer::EvaluationLayer, st) = optimization_schedule(layer.shooting, st.shooting)
optimization_state(layer::EvaluationLayer, st, grid) = merge(st, (; shooting = optimization_state(layer.shooting, st.shooting, grid)))
optimization_prediction(layer::EvaluationLayer, ps, st, grid) = optimization_prediction(layer.shooting, ps.shooting, st.shooting, grid)
get_number_of_shooting_constraints(layer::EvaluationLayer, ps, st) =
    get_number_of_shooting_constraints(layer.shooting, ps.shooting, st.shooting)
trajectory_objective(layer::EvaluationLayer, loss) = trajectory_objective(layer.shooting, loss)

shooting_components(layer, ps, st) = (layer, ps, st)
shooting_components(layer::EvaluationLayer, ps, st) = (layer.shooting, ps.shooting, st.shooting)

"""
    residual(layer::EvaluationLayer, experiment::Experiment, trajectory, parameters; evaluation_state, eval_module, evaluator)

Evaluate observations against saved samples. Pass the current named values
from `ps.evaluation` for observation-only tunables. Use `evaluation_state` to
supply fixed values from Lux state when they differ from the layer defaults.
"""
function residual(
        layer::EvaluationLayer, experiment::Experiment, trajectory, parameters;
        eval_module::Module = @__MODULE__, evaluator = default_evaluator(), evaluation_state = layer.fixed
    )
    return residual_from_samples(layer, experiment, trajectory, evaluation_values(layer, parameters, evaluation_state), eval_module, evaluator)
end
