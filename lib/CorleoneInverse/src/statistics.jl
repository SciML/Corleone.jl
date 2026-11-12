"""
$(TYPEDEF)

Asymptotic statistics of a least-squares fit in natural parameter coordinates,
in the packed order of the Lux parameters.

# Fields
$(FIELDS)
"""
@concrete terse struct ParameterEstimate
    "The natural parameter values."
    x
    "The covariance matrix."
    covariance
    "The standard errors."
    std_errors
    "The `(lower, upper)` confidence interval of each parameter."
    confidence_intervals
    "The confidence level `1 - α`."
    level
    "The observation residual sum of squares."
    rss
    "The residual degrees of freedom."
    dof
end

"""
$(SIGNATURES)

Construct `ParameterEstimate(model, q; ps = model.ps, st = model.st)` for a prepared
experiment dynamic problem. The convenience form
`ParameterEstimate(shooting, experiments, q, ps, st)` prepares the model first.
The covariance in internal coordinates is `(JᵀJ / s² + H)⁻¹` for the observation
residual Jacobian `J` and the Hessian `H` of the negative [`log_prior`](@ref),
restricted to the null space of equality constraints, including shooting matches.
The approximation assumes an interior optimum relative to inequalities and bounds. With
known noise scales `s² = 1` and intervals use normal quantiles; otherwise `s²` is
the residual variance and intervals use Student-t quantiles; priors then require
known noise scales. Values are reported in
natural coordinates of the annotation transforms: the covariance with the delta
method, the intervals by mapping their internal endpoints.

With priors, supply `q` from a fit that minimizes half the weighted residual sum
of squares minus the log prior. Use `NonlinearLeastSquaresProblem` for supported
Normal priors or construct the model with `reducer = x -> sum(x) / 2` for
`OptimizationProblem`. The default mean-squared objective changes the relative
weight of the prior. `ParameterEstimate` uses the Gaussian likelihood convention
for every reducer.
"""
function ParameterEstimate(
        model::ExperimentDynamicOptProblem, q::AbstractVector;
        ps = model.ps, st = model.st, α::Real = 0.05, vectorize = default_vectorizer()
    )
    0 < α < 1 || throw(ArgumentError("α must lie in (0, 1)."))
    owner = model.objective.owner
    _, reconstruct = pack(vectorize, ps)
    natural = x -> natural_vector(owner, reconstruct(x), st)
    observation_residuals = x -> residual(model, reconstruct(x), st)
    r = observation_residuals(q)
    J = ForwardDiff.jacobian(observation_residuals, q)
    H = ForwardDiff.hessian(x -> -log_prior(owner, reconstruct(x), st), q)
    equality_rows = findall(eachindex(model.lcons)) do i
        isfinite(model.lcons[i]) && model.lcons[i] == model.ucons[i]
    end
    Z = if isempty(equality_rows)
        Matrix(I, length(q), length(q))
    else
        C = ForwardDiff.jacobian(x -> optimization_constraints(model, reconstruct(x), st)[equality_rows], q)
        nullspace(C)
    end
    rss = sum(abs2, r)
    dof = length(r) - size(Z, 2)
    dof > 0 || throw(ArgumentError("Need more observations than free parameters, got $(length(r)) for $(size(Z, 2))."))
    weighted = first(is_weighted(first(values(model.objective.data))))
    # The fit weights a prior against unit noise; the estimated s² would rescale only the data.
    weighted || all(entry -> isnothing(first(entry).prior), parameter_collections(owner, ps, st)) ||
        throw(ArgumentError("Parameter statistics with priors require observations with a noise scale σ."))
    s² = weighted ? one(rss) : rss / dof
    Σφ = Z * inv(Symmetric(Z' * (J' * J / s² + H) * Z)) * Z'
    G = ForwardDiff.jacobian(natural, q)
    Σ = G * Σφ * G'
    z = quantile(weighted ? Normal() : TDist(dof), 1 - α / 2)
    halfwidth = z .* sqrt.(diag(Σφ))
    intervals = collect(zip(natural(q .- halfwidth), natural(q .+ halfwidth)))
    return ParameterEstimate(natural(q), Σ, sqrt.(diag(Σ)), intervals, 1 - α, rss, dof)
end

function ParameterEstimate(
        shooting::ShootingLayer, experiments::Union{ObservationExperiments, Experiments}, q::AbstractVector, ps, st;
        α::Real = 0.05, vectorize = default_vectorizer(), eval_module::Module = Corleone,
        evaluator = Corleone.default_evaluator(), observation_parameters::NamedTuple = (;)
    )
    model = CorleoneDynamicOptProblem(shooting, experiments; ps, st, eval_module, evaluator, observation_parameters)
    return ParameterEstimate(model, q; α, vectorize)
end

# The natural values of the tunable parameters, in packed order.
natural_vector(layer, ps, st) = reduce(
    vcat, map(((collection, values),) -> natural_values(collection.transform, values.ps), parameter_collections(layer, ps, st));
    init = Union{}[]
)

function Base.show(io::IO, ::MIME"text/plain", est::ParameterEstimate)
    println(io, "ParameterEstimate (RSS = ", round(est.rss; sigdigits = 4), ", dof = ", est.dof, ")")
    print(io, "  index  value  std. error  ", round(100 * est.level; digits = 1), "% interval")
    for (i, (x, se, (lo, hi))) in enumerate(zip(est.x, est.std_errors, est.confidence_intervals))
        print(
            io, "\n  ", i, "  ", round(x; sigdigits = 5), "  ", round(se; sigdigits = 3), "  [",
            round(lo; sigdigits = 5), ", ", round(hi; sigdigits = 5), "]"
        )
    end
    return nothing
end
