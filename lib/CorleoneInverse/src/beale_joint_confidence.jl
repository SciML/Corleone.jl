"""
    beale_joint_confidence(model, estimate; kwargs...)

Sample a joint parameter confidence region using Beale's method, based on a
[`ParameterEstimate`](@ref). Each parameter is sampled uniformly in a box
centered on its 95% confidence interval. A sample is accepted when its sum of
squared residuals is below
`SSR_min * (1 + p / (n - p) * F_quantile(p, n - p; confidence))`.

The confidence-interval widths are scaled by `expansion`. Parameters can be
restricted to the sign of their estimate with `sign_restricted`.

Keyword arguments:
- `names`: names used for each parameter (default: `θ1`, `θ2`, ...).
- `sign_restricted`: whether to keep each parameter on the sign of its estimate.
- `allow_zero`: whether samples exactly equal to zero are allowed.
- `expansion`: sampling width relative to the 95% interval width (default: 2).
- `n_accept`: number of accepted combinations to collect (default: 500).
- `max_samples`: maximum number of combinations to test (default: 100000).
- `confidence`: confidence level for the F-test (default: 0.95).
- `rng`: random number generator.

Scalar `sign_restricted`, `allow_zero`, and `expansion` values apply to every
parameter; collections provide per-parameter values.

Returns a named tuple containing `parameters` (a parameter-by-sample matrix),
the accepted `ssr` values, threshold, minimum SSR, estimate, sampling limits,
parameter names, number of tested samples, and confidence level.

Contributed by Dr. Kai Exner, BASF SE.
"""
function beale_joint_confidence(
        model, estimate::ParameterEstimate;
        names = ["θ$i" for i in eachindex(estimate.x)],
        sign_restricted = false, allow_zero = true, expansion = 2.0,
        n_accept::Integer = 500, max_samples::Integer = 100_000,
        confidence::Real = 0.95, rng::Random.AbstractRNG = Random.default_rng()
    )
    p = length(estimate.x)
    names = collect(String, names)
    length(names) == p || throw(ArgumentError("Expected $p names, got $(length(names))."))
    0 < confidence < 1 || throw(ArgumentError("confidence must lie in (0, 1)."))
    n_accept > 0 || throw(ArgumentError("n_accept must be positive."))
    max_samples >= 0 || throw(ArgumentError("max_samples must be nonnegative."))
    length(estimate.confidence_intervals) == p ||
        throw(ArgumentError("Expected $p confidence intervals, got $(length(estimate.confidence_intervals))."))

    restricted = _beale_per_parameter(sign_restricted, p, "sign_restricted")
    zero_ok = _beale_per_parameter(allow_zero, p, "allow_zero")
    factors = _beale_per_parameter(expansion, p, "expansion")
    all(x -> x isa Bool, restricted) ||
        throw(ArgumentError("sign_restricted must contain only Bool values."))
    all(x -> x isa Bool, zero_ok) ||
        throw(ArgumentError("allow_zero must contain only Bool values."))
    all(x -> x isa Real && isfinite(x) && x > 0, factors) ||
        throw(ArgumentError("expansion must contain only positive finite values."))

    owner, ps, st = model.objective.owner, model.ps, model.st
    q0, reconstruct = pack(default_vectorizer(), ps)
    length(q0) == p ||
        throw(ArgumentError("The estimate has $p parameters, but the model has $(length(q0)) tunable values."))
    natural = q -> natural_vector(owner, reconstruct(q), st)
    transforms = _beale_transforms(owner, ps, st)
    # Independent box samples cannot follow equalities such as shooting matches.
    any(i -> model.lcons[i] == model.ucons[i], eachindex(model.lcons)) &&
        throw(ArgumentError("Beale sampling does not support equality constraints, including multiple-shooting matching conditions."))
    # The region is defined by the residual sum of squares alone.
    all(entry -> isnothing(first(entry).prior), parameter_collections(owner, ps, st)) ||
        throw(ArgumentError("Beale sampling does not support parameter priors."))
    constrained = any(isfinite, model.lcons) || any(isfinite, model.ucons)
    lower_bounds, upper_bounds = get_bounds(model.layer, ps, st)
    lb, _ = pack(default_vectorizer(), lower_bounds)
    ub, _ = pack(default_vectorizer(), upper_bounds)

    # Sample in internal coordinates, where every value lies in the transform's domain.
    # The natural intervals map from internal endpoints, so they map back exactly.
    estimate_values = copy(estimate.x)
    center = internal_vector(owner, ps, st, estimate_values)
    interval_lower = internal_vector(owner, ps, st, first.(estimate.confidence_intervals))
    interval_upper = internal_vector(owner, ps, st, last.(estimate.confidence_intervals))
    lower = zeros(p)
    upper = zeros(p)
    for i in 1:p
        lo, hi = minmax(interval_lower[i], interval_upper[i])
        isfinite(lo) && isfinite(hi) ||
            throw(ArgumentError("Invalid confidence interval for $(names[i])."))
        halfwidth = factors[i] * (hi - lo) / 2
        lower[i] = center[i] - halfwidth
        upper[i] = center[i] + halfwidth
        if restricted[i]
            estimate_values[i] == 0 &&
                throw(ArgumentError("Cannot sign-restrict $(names[i]): its estimate is zero."))
            zero_point = _beale_internal_zero(transforms[i])
            # Keep the side of the internal zero that holds the estimate.
            if !isnothing(zero_point)
                zero_point < center[i] ? (lower[i] = max(lower[i], zero_point)) :
                    (upper[i] = min(upper[i], zero_point))
            end
        end
        # Annotation bounds are stored in internal coordinates.
        lower[i] = max(lower[i], lb[i])
        upper[i] = min(upper[i], ub[i])
        lower[i] < upper[i] || throw(ArgumentError("Empty sampling interval for $(names[i])."))
    end

    n = length(residual(model, ps, st))
    # Without equalities, every tunable value is a free parameter of the fit.
    n - estimate.dof == p ||
        throw(ArgumentError("The estimate has $(n - estimate.dof) free parameters for $n residuals, but the model has $p."))
    minimal_ssr = estimate.rss
    # Known noise scales give a likelihood-ratio region; otherwise the variance is
    # estimated from the residuals and the F distribution accounts for it.
    weighted = first(is_weighted(first(values(model.objective.data))))
    threshold = if weighted
        minimal_ssr + quantile(Chisq(p), confidence)
    else
        minimal_ssr * (1 + p / estimate.dof * quantile(FDist(p, estimate.dof), confidence))
    end

    accepted = Vector{Float64}[]
    accepted_ssr = Float64[]
    tested = 0
    while length(accepted) < n_accept && tested < max_samples
        tested += 1
        q = lower .+ (upper .- lower) .* rand(rng, p)
        θ = natural(q)
        any(i -> θ[i] == 0 && !zero_ok[i], 1:p) && continue
        candidate = reconstruct(q)
        constrained && !_beale_feasible(model, candidate) && continue
        ssr = _beale_ssr(model, candidate)
        if ssr < threshold
            push!(accepted, θ)
            push!(accepted_ssr, ssr)
        end
    end
    length(accepted) < n_accept &&
        @warn "Only $(length(accepted)) of $n_accept requested combinations accepted after $tested samples."

    parameters = isempty(accepted) ? zeros(p, 0) : reduce(hcat, accepted)
    return (
        parameters = parameters, ssr = accepted_ssr, threshold = threshold,
        minimal_ssr = minimal_ssr, estimate = estimate_values,
        lower = min.(natural(lower), natural(upper)), upper = max.(natural(lower), natural(upper)),
        names = names, tested = tested, confidence = confidence,
    )
end

function _beale_per_parameter(value::Union{Bool, Real}, p, name)
    result = fill(value, p)
    return result
end

function _beale_per_parameter(values, p, name)
    result = collect(values)
    length(result) == p ||
        throw(ArgumentError("$name must have one value per parameter (expected $p, got $(length(result)))."))
    return result
end

# The transform of each packed coordinate.
_beale_transforms(owner, ps, st) = reduce(
    vcat, map(((collection, values),) -> fill(collection.transform, length(values.ps)), collect(parameter_collections(owner, ps, st)));
    init = Any[]
)

# The internal coordinate of a natural zero, or `nothing` when zero lies outside the transform's image.
function _beale_internal_zero(transform)
    try
        return only(internal_values(transform, [0.0]))
    catch error
        error isa DomainError && return nothing
        rethrow()
    end
end

# Whether a candidate satisfies the model's inequality constraints.
function _beale_feasible(model, ps)
    try
        c = optimization_constraints(model, ps, model.st)
        return all(i -> model.lcons[i] <= c[i] <= model.ucons[i], eachindex(c))
    catch error
        error isa DomainError && return false
        rethrow()
    end
end

function _beale_ssr(model, ps)
    try
        ssr = sum(abs2, residual(model, ps, model.st))
        # Failed solves yield non-finite residuals and reject the sample.
        return isfinite(ssr) ? ssr : Inf
    catch error
        # A candidate outside the model's domain is rejected, as in Beale's method;
        # other errors, including interrupts, propagate.
        error isa DomainError && return Inf
        rethrow()
    end
end

"""
    plot_beale_pairs(result; pairs=nothing, show_sampling_box=true, columns=2)

Plot accepted Beale confidence-region samples pairwise. `pairs` may contain
parameter names or indices; by default all parameter pairs are plotted. Sampling
limits are shown as dashed red lines and the estimate as a black star. Requires
`CairoMakie` to be loaded.
"""
function plot_beale_pairs end
