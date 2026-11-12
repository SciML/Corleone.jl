# Gaussian priors on untransformed values are equivalent to residual rows; other
# priors need the log density and are only supported by `OptimizationProblem`.
# Values duplicated by `add_parameter_segments!` contribute no additional rows.
prior_residuals(layer, ps, st) = reduce(
    vcat, map(parameter_collections(layer, ps, st)) do (collection, values, inserted)
        x = values.ps
        prior_residual(collection.prior, collection.transform, x[setdiff(eachindex(x), inserted)])
    end;
    init = Float64[]
)
prior_residual(::Nothing, transform, x) = similar(x, 0)
prior_residual(prior::Normal, ::BoundedTransform, x) = (x .- mean(prior)) ./ std(prior)
prior_residual(prior, transform, x) = throw(
    ArgumentError(
        "Least-squares problems support Normal priors on untransformed parameters only; use OptimizationProblem."
    )
)

"""
$(SIGNATURES)

Construct a residual least-squares problem from an experiment
[`CorleoneDynamicOptProblem`](@ref). The prepared observation maps and sampling
indices are reused. Untransformed Normal priors add residual rows. Finite
parameter bounds are passed as `lb` and `ub` in internal coordinates; NonlinearSolve
honors them natively or through a bounds transform. General priors and constraints
use `OptimizationProblem(model, ad)`.

`ps` supplies the initial unknowns and `st` becomes the SciMLBase problem's
parameter `p`. Both default to the model's retained values. A replacement
state must match the prepared schedules and layouts. Remaining keywords are
forwarded to the SciMLBase problem.
"""
function SciMLBase.NonlinearLeastSquaresProblem(
        model::ExperimentDynamicOptProblem; ps = model.ps, st = model.st,
        vectorize = default_vectorizer(), kwargs...
    )
    all(==(-Inf), model.lcons) && all(==(Inf), model.ucons) ||
        throw(ArgumentError("Least-squares problems require an unconstrained model; use OptimizationProblem."))
    owner = model.objective.owner
    q0, reconstruct = pack(vectorize, ps)
    n_prior = length(prior_residuals(owner, ps, st))
    f = if iszero(n_prior)
        (q, state) -> residual(model, reconstruct(q), isnothing(state) ? st : state)
    else
        (q, state) -> begin
            parameters = reconstruct(q)
            current = isnothing(state) ? st : state
            vcat(residual(model, parameters, current), prior_residuals(owner, parameters, current))
        end
    end
    prototype = zeros(eltype(q0), n_observations(model) + n_prior)
    lower, upper = get_bounds(model.layer, ps, st)
    lb, _ = pack(vectorize, lower)
    ub, _ = pack(vectorize, upper)
    # Unbounded problems keep the solver's default unconstrained algorithm.
    bounded = any(isfinite, lb) || any(isfinite, ub)
    return SciMLBase.NonlinearLeastSquaresProblem(
        SciMLBase.NonlinearFunction{false}(f; resid_prototype = prototype), q0, st;
        lb = bounded ? lb : nothing, ub = bounded ? ub : nothing, kwargs...
    )
end

"""
$(SIGNATURES)

Prepare an experiment dynamic problem and construct its least-squares problem.
This convenience form retains the parameter layout of `ps` and `st`.
"""
function SciMLBase.NonlinearLeastSquaresProblem(
        shooting::ShootingLayer, experiments::Union{ObservationExperiments, Experiments}, ps, st;
        vectorize = default_vectorizer(), eval_module::Module = Corleone,
        evaluator = Corleone.default_evaluator(), observation_parameters::NamedTuple = (;), kwargs...
    )
    model = CorleoneDynamicOptProblem(shooting, experiments; ps, st, eval_module, evaluator, observation_parameters)
    return SciMLBase.NonlinearLeastSquaresProblem(model; vectorize, kwargs...)
end

"""
$(SIGNATURES)

Prepare an experiment dynamic problem and use Corleone's common Optimization.jl
bridge for its observation loss, priors, bounds, and shooting constraints.
`reducer` selects the scalar reduction of the squared observation residuals. Additional
user constraints can be declared with `CorleoneDynamicOptProblem` directly.
"""
function SciMLBase.OptimizationProblem(
        shooting::ShootingLayer, experiments::Union{ObservationExperiments, Experiments}, ps, st, ad::AbstractADType;
        vectorize = default_vectorizer(), eval_module::Module = Corleone,
        evaluator = Corleone.default_evaluator(), reducer = mean, observation_parameters::NamedTuple = (;), kwargs...
    )
    model = CorleoneDynamicOptProblem(shooting, experiments; ps, st, eval_module, evaluator, reducer, observation_parameters)
    return SciMLBase.OptimizationProblem(model, ad; vectorize, kwargs...)
end

function appended_experiment_model(model, experiment, u0, ps, st, vectorize, rng)
    prediction = model.layer
    owner = prediction.layer
    owner isa Experiments || throw(ArgumentError("Use an Experiments layer to append to an inverse model."))
    initial, reconstruct = pack(vectorize, ps)
    length(u0) == length(initial) || throw(DimensionMismatch("The warm start must match the existing model's packed layout."))
    fitted = reconstruct(u0)
    observation_parameters = isnothing(prediction.evaluation) ? (;) :
        NamedTuple{keys(prediction.evaluation.evaluation)}(evaluation_parameters(prediction.evaluation, fitted))
    next = append_experiment(owner, experiment_parameters(prediction, fitted), st, experiment; rng)
    definition = prediction.definition
    constraints = map(definition.constraints) do constraint
        # An untagged symbolic constraint originally selected the sole experiment.
        constraint isa Expr ? only(keys(owner.experiments)) => constraint : constraint
    end
    return CorleoneDynamicOptProblem(
        prediction.shooting, next.layer, constraints...;
        next.ps, next.st, definition.lcons, definition.ucons,
        definition.eval_module, definition.evaluator, model.objective.reducer, observation_parameters
    )
end

"""
    append_experiment(model, problem, experiment; u0 = problem.u0, st = problem.p, kwargs...)

Return `(; model, problem)` with the named experiment appended to a prepared
inverse model and its SciML solver problem. Pass `u0 = fit.u` to warm-start
from a completed fit. Rebuild solver callbacks, bounds, and the residual or
constraint layout. Use the same `vectorize` implementation as the original
problem and supply `ps` if it was built with a different parameter layout.
Extra keywords configure the rebuilt solver bridge.
"""
function append_experiment(
        model::ExperimentDynamicOptProblem, problem::SciMLBase.OptimizationProblem, experiment::Pair{Symbol};
        u0 = problem.u0, ps = model.ps, st = problem.p,
        vectorize = default_vectorizer(), rng = Random.default_rng(), kwargs...
    )
    isnothing(problem.int) || throw(ArgumentError("Rebuild integer masks explicitly when appending experiments."))
    next = appended_experiment_model(model, experiment, u0, ps, st, vectorize, rng)
    prepared = SciMLBase.OptimizationProblem(next, problem.f.adtype; vectorize, inplace = SciMLBase.isinplace(problem), kwargs...)
    bounded = !isnothing(problem.lb)
    updated = SciMLBase.remake(
        problem; prepared.f, prepared.u0, prepared.p, prepared.lcons, prepared.ucons,
        lb = bounded ? prepared.lb : nothing, ub = bounded ? prepared.ub : nothing,
        prepared.kwargs...
    )
    return (; model = next, problem = updated)
end

function append_experiment(
        model::ExperimentDynamicOptProblem, problem::SciMLBase.NonlinearLeastSquaresProblem, experiment::Pair{Symbol};
        u0 = problem.u0, ps = model.ps, st = problem.p,
        vectorize = default_vectorizer(), rng = Random.default_rng(), kwargs...
    )
    next = appended_experiment_model(model, experiment, u0, ps, st, vectorize, rng)
    options = merge((; problem.kwargs...), (; kwargs...))
    updated = SciMLBase.NonlinearLeastSquaresProblem(next; vectorize, options...)
    return (; model = next, problem = updated)
end
