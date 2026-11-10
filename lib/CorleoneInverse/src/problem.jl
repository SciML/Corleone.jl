function experiment_states(layer, experiments, st)
    isempty(experiments) && throw(ArgumentError("Supply at least one experiment."))
    weighted = reduce(vcat, map(collect ∘ is_weighted, values(experiments)))
    all(weighted) || !any(weighted) ||
        throw(ArgumentError("Either all observations or none must specify a noise scale σ."))
    # The experiments share `ps`, which would include their shooting node values.
    length(experiments) == 1 || get_number_of_shooting_constraints(layer, nothing, st) == 0 ||
        throw(ArgumentError("Multiple shooting is supported for a single experiment only."))
    return map(experiment -> experiment_state(layer, experiment, st), experiments)
end

# Pack the parameters once and prepare the per-experiment states. `predict` maps
# packed parameters to the experiments' trajectories.
function prepare(layer, experiments, ps, st, vectorize)
    experiments = as_experiments(experiments)
    states = experiment_states(layer, experiments, st)
    q0, reconstruct = pack(vectorize, ps)
    predict = let layer = layer, experiments = experiments, states = states, reconstruct = reconstruct
        q -> map((experiment, state) -> first(layer(experiment, reconstruct(q), state)), experiments, states)
    end
    return (; experiments, states, q0, reconstruct, predict)
end

matching(trajectories) = reduce(vcat, map(shooting_constraints, values(trajectories)))
n_matching(layer, ps, states) = sum(state -> get_number_of_shooting_constraints(layer, ps, state), values(states))

# Gaussian priors on untransformed values are equivalent to residual rows; other
# priors need the log density and are only supported by `OptimizationProblem`.
prior_residuals(layer, ps) = reduce(
    vcat, map(((collection, values),) -> prior_residual(collection.prior, collection.transform, values.ps), parameter_collections(layer, ps));
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

Build a `NonlinearLeastSquaresProblem` fitting the tunable parameters of `layer` to
`experiments` (an [`Experiment`](@ref), tuple or NamedTuple), starting from `ps`
with base state `st`. The unknowns are the packed Lux parameters, in the internal
coordinates of any annotation transforms. Normal priors on untransformed parameters
add the residuals `(θ - μ) / σ`. Bounds are not passed; use annotation transforms, or
`OptimizationProblem` for bounds, other priors, and multiple shooting. Remaining
keywords are forwarded to the problem.
"""
function SciMLBase.NonlinearLeastSquaresProblem(
        layer::ShootingLayer, experiments::Experiments, ps, st; vectorize = default_vectorizer(), kwargs...
    )
    (; experiments, states, q0, reconstruct, predict) = prepare(layer, experiments, ps, st, vectorize)
    n_matching(layer, ps, states) == 0 ||
        throw(ArgumentError("Multiple shooting requires matching constraints; use OptimizationProblem."))
    n_prior = length(prior_residuals(layer, ps))
    f = let layer = layer, experiments = experiments, predict = predict, reconstruct = reconstruct
        (q, _) -> vcat(residual(experiments, predict(q)), prior_residuals(layer, reconstruct(q)))
    end
    resid_prototype = zeros(eltype(q0), n_observations(experiments) + n_prior)
    return SciMLBase.NonlinearLeastSquaresProblem(
        SciMLBase.NonlinearFunction{false}(f; resid_prototype), q0; kwargs...
    )
end

"""
$(SIGNATURES)

Build an `OptimizationProblem` minimizing half the residual sum of squares minus
[`log_prior`](@ref), with the layer's bounds and multiple-shooting matching
constraints. Keywords follow `NonlinearLeastSquaresProblem`; remaining keywords are
forwarded to the problem.
"""
function SciMLBase.OptimizationProblem(
        layer::ShootingLayer, experiments::Experiments, ps, st, ad::AbstractADType;
        vectorize = default_vectorizer(), kwargs...
    )
    (; experiments, states, q0, reconstruct, predict) = prepare(layer, experiments, ps, st, vectorize)
    lower, upper = get_bounds(layer, ps, st)
    lb, ub = first(pack(vectorize, lower)), first(pack(vectorize, upper))
    objective = let layer = layer, experiments = experiments, predict = predict, reconstruct = reconstruct
        (q, _) -> sum(abs2, residual(experiments, predict(q))) / 2 - log_prior(layer, reconstruct(q))
    end
    n = n_matching(layer, ps, states)
    cons = n == 0 ? nothing : let predict = predict
            (res, q, _) -> copyto!(res, matching(predict(q)))
    end
    f = SciMLBase.OptimizationFunction{true}(objective, ad; cons)
    lcons, ucons = n == 0 ? (nothing, nothing) : (zeros(n), zeros(n))
    return SciMLBase.OptimizationProblem(f, q0; lb, ub, lcons, ucons, kwargs...)
end
