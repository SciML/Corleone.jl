struct EvaluationSparsity
    jacobian::SparseMatrixCSC{Bool, Int}
    hessian::SparseMatrixCSC{Bool, Int}
end

"""
    EvaluationLayer(shooting::ShootingLayer, objective, constraints...; parameters...)

Wrap a shooting layer with additional scalar trainable parameters for
[`CorleoneDynamicOptProblem`](@ref) expressions. Pass the objective first,
then the constraints in declaration order. For example, prepare
`EvaluationLayer(shooting, :(gain * u₁(1.0)), :gain; gain = 1.0)` with
`CorleoneDynamicOptProblem(layer; lcons = 0.0, ucons = Inf)`.

The parameters are stored in `ps.evaluation` and the shooting layer receives
`ps.shooting`. Names must be distinct from shooting signals. Additional
parameters are continuous and unbounded.

The prepared problem's `layer.sparsity` stores the packed constraint Jacobian
and Lagrangian Hessian patterns as `jacobian` and `hessian`. For ODE shooting,
preparation fills this snapshot when the default vectorizer is loaded. Otherwise
the first packed sparsity query fills it. The snapshot uses the packing at
preparation time.
"""
mutable struct EvaluationLayer{S <: ShootingLayer, O, C <: Tuple, P <: NamedTuple} <: LuxCore.AbstractLuxLayer
    const shooting::S
    const objective::O
    const constraints::C
    const evaluation::P
    sparsity::Union{Nothing, EvaluationSparsity}

    function EvaluationLayer(shooting::ShootingLayer, objective, constraints...; parameters...)
        defaults = (; parameters...)
        isempty(defaults) && throw(ArgumentError("Supply at least one evaluation parameter."))
        all(x -> x isa Real && !(x isa Bool) && isfinite(x), defaults) ||
            throw(ArgumentError("Evaluation parameter defaults must be finite real scalars."))
        signals = expression_symbol_indices(shooting)
        for name in keys(defaults)
            haskey(signals, name) && throw(ArgumentError("Evaluation parameter $name duplicates a shooting signal."))
        end
        values = map(float, defaults)
        return new{typeof(shooting), typeof(objective), typeof(constraints), typeof(values)}(shooting, objective, constraints, values, nothing)
    end

    # Each prepared problem owns its packed patterns.
    EvaluationLayer(layer::EvaluationLayer{S, O, C, P}) where {S, O, C, P} =
        new{S, O, C, P}(layer.shooting, layer.objective, layer.constraints, layer.evaluation, nothing)
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, layer::EvaluationLayer)
    return (; shooting = LuxCore.initialparameters(rng, layer.shooting), evaluation = layer.evaluation)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::EvaluationLayer)
    return (; shooting = LuxCore.initialstates(rng, layer.shooting), evaluation = (;))
end

LuxCore.parameterlength(layer::EvaluationLayer) = LuxCore.parameterlength(layer.shooting) + length(layer.evaluation)
LuxCore.statelength(layer::EvaluationLayer) = LuxCore.statelength(layer.shooting)

function (layer::EvaluationLayer)(x, ps, st)
    traj, shooting = LuxCore.apply(layer.shooting, x, ps.shooting, st.shooting)
    return traj, (; shooting, evaluation = st.evaluation)
end

for operation in (:get_lower_bound, :get_upper_bound, :get_integrality)
    @eval function $operation(layer::EvaluationLayer, ps, st = nothing)
        shooting = $operation(layer.shooting, ps.shooting, bound_child_state(st, :shooting))
        evaluation = $operation(nothing, ps.evaluation)
        return evaluation_parameter_tree(ps, shooting, evaluation)
    end
end

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

evaluation_parameters(layer, ps) = ()
evaluation_parameters(layer::EvaluationLayer, ps) = evaluation_values(layer, ps.evaluation)
function evaluation_values(layer::EvaluationLayer, values)
    return map(keys(layer.evaluation)) do name
        hasproperty(values, name) || throw(ArgumentError("Missing evaluation parameter $name."))
        getproperty(values, name)
    end
end
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
    residual(layer::EvaluationLayer, experiment::Experiment, trajectory, parameters; eval_module, evaluator)

Evaluate observations against saved samples. Pass the current named values
from `ps.evaluation` for observation-only parameters.
"""
function residual(
        layer::EvaluationLayer, experiment::Experiment, trajectory, parameters;
        eval_module::Module = @__MODULE__, evaluator = default_evaluator()
    )
    return residual_from_samples(layer, experiment, trajectory, evaluation_values(layer, parameters), eval_module, evaluator)
end
