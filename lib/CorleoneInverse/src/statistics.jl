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

Construct a [`ParameterEstimate`](@ref) for the fitted packed parameters `q` of
`layer` and `experiments`; `ps` and `st` must be those used to build the problem.
The covariance in internal coordinates is `(JᵀJ / s² + H)⁻¹` for the observation
residual Jacobian `J` and the Hessian `H` of the negative [`log_prior`](@ref),
restricted to the null space of the multiple-shooting matching constraints. With
known noise scales `s² = 1` and intervals use normal quantiles; otherwise `s²` is
the residual variance and intervals use Student-t quantiles. Values are reported in
natural coordinates of the annotation transforms: the covariance with the delta
method, the intervals by mapping their internal endpoints.
"""
function ParameterEstimate(
        layer::ShootingLayer, experiments::Union{ObservationExperiments, Experiments}, q::AbstractVector, ps, st;
        α::Real = 0.05, vectorize = default_vectorizer()
    )
    0 < α < 1 || throw(ArgumentError("α must lie in (0, 1)."))
    owner = experiments isa Experiments ? experiments : layer
    (; experiments, states, reconstruct, predict) = prepare(layer, experiments, ps, st, vectorize)
    natural = x -> natural_vector(owner, reconstruct(x))
    r = residual(experiments, predict(q))
    J = ForwardDiff.jacobian(x -> residual(experiments, predict(x)), q)
    H = ForwardDiff.hessian(x -> -log_prior(owner, reconstruct(x)), q)
    Z = n_matching(layer, ps, states) == 0 ? Matrix(I, length(q), length(q)) :
        nullspace(ForwardDiff.jacobian(x -> matching(predict(x)), q))
    rss = sum(abs2, r)
    dof = length(r) - size(Z, 2)
    dof > 0 || throw(ArgumentError("Need more observations than free parameters, got $(length(r)) for $(size(Z, 2))."))
    weighted = first(is_weighted(first(experiments)))
    s² = weighted ? one(rss) : rss / dof
    Σφ = Z * inv(Symmetric(Z' * (J' * J / s² + H) * Z)) * Z'
    G = ForwardDiff.jacobian(natural, q)
    Σ = G * Σφ * G'
    z = quantile(weighted ? Normal() : TDist(dof), 1 - α / 2)
    halfwidth = z .* sqrt.(diag(Σφ))
    intervals = collect(zip(natural(q .- halfwidth), natural(q .+ halfwidth)))
    return ParameterEstimate(natural(q), Σ, sqrt.(diag(Σ)), intervals, 1 - α, rss, dof)
end

# The natural values of the tunable parameters, in packed order.
natural_vector(layer, ps) = reduce(
    vcat, map(((collection, values),) -> natural_values(collection.transform, values.ps), parameter_collections(layer, ps))
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
