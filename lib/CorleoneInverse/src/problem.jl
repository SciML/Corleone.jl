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

`ps` and `st` default to the model's retained values. Remaining keywords are
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
    f = let model = model, owner = owner, st = st, reconstruct = reconstruct
        (q, _) -> begin
            parameters = reconstruct(q)
            vcat(residual(model, parameters, st), prior_residuals(owner, parameters, st))
        end
    end
    prototype = zeros(eltype(q0), n_observations(model) + n_prior)
    lower, upper = get_bounds(model.layer, ps, st)
    lb, _ = pack(vectorize, lower)
    ub, _ = pack(vectorize, upper)
    # Unbounded problems keep the solver's default unconstrained algorithm.
    bounded = any(isfinite, lb) || any(isfinite, ub)
    return SciMLBase.NonlinearLeastSquaresProblem(
        SciMLBase.NonlinearFunction{false}(f; resid_prototype = prototype), q0;
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
        evaluator = Corleone.default_evaluator(), kwargs...
    )
    model = CorleoneDynamicOptProblem(shooting, experiments; ps, st, eval_module, evaluator)
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
        evaluator = Corleone.default_evaluator(), reducer = mean, kwargs...
    )
    model = CorleoneDynamicOptProblem(shooting, experiments; ps, st, eval_module, evaluator, reducer)
    return SciMLBase.OptimizationProblem(model, ad; vectorize, kwargs...)
end
