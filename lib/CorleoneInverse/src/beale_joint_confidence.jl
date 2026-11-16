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

    estimate_values = copy(estimate.x)
    lower = zeros(p)
    upper = zeros(p)
    for i in 1:p
        lo, hi = estimate.confidence_intervals[i]
        isfinite(lo) && isfinite(hi) && lo <= hi ||
            throw(ArgumentError("Invalid confidence interval for $(names[i])."))
        midpoint = (lo + hi) / 2
        halfwidth = factors[i] * (hi - lo) / 2
        lower[i] = midpoint - halfwidth
        upper[i] = midpoint + halfwidth
        if restricted[i]
            estimate_values[i] == 0 &&
                throw(ArgumentError("Cannot sign-restrict $(names[i]): its estimate is zero."))
            estimate_values[i] > 0 ? (lower[i] = max(lower[i], 0.0)) : (upper[i] = min(upper[i], 0.0))
        end
        lower[i] < upper[i] || throw(ArgumentError("Empty sampling interval for $(names[i])."))
    end

    _, reconstruct = Corleone.pack(Corleone.default_vectorizer(), model.ps)
    n = length(residual(model, model.ps, model.st))
    n > p || throw(ArgumentError("Need more residuals than parameters."))
    minimal_ssr = estimate.rss
    threshold = minimal_ssr * (1 + p / (n - p) * quantile(FDist(p, n - p), confidence))

    accepted = Vector{Float64}[]
    accepted_ssr = Float64[]
    tested = 0
    while length(accepted) < n_accept && tested < max_samples
        tested += 1
        θ = lower .+ (upper .- lower) .* rand(rng, p)
        any(i -> θ[i] == 0 && !zero_ok[i], 1:p) && continue
        ssr = _beale_ssr(model, reconstruct, θ)
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
        minimal_ssr = minimal_ssr, estimate = estimate_values, lower = lower, upper = upper,
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

function _internal_beale_vector(owner, ps, st, θ)
    parts = Vector{Float64}[]
    offset = 0
    for entry in Corleone.parameter_collections(owner, ps, st)
        collection, values = entry[1], entry[2]
        count = length(Corleone.natural_values(collection.transform, values.ps))
        push!(parts, Corleone.internal_values(collection.transform, θ[(offset + 1):(offset + count)]))
        offset += count
    end
    offset == length(θ) ||
        throw(ArgumentError("The sampled parameter vector does not match the model's tunable parameters."))
    return reduce(vcat, parts)
end

function _beale_ssr(model, reconstruct, θ)
    q = _internal_beale_vector(model.objective.owner, model.ps, model.st, θ)
    try
        ssr = sum(abs2, residual(model, reconstruct(q), model.st))
        return isfinite(ssr) ? ssr : Inf
    catch
        # A failed model evaluation rejects this sample, as in Beale's method.
        return Inf
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
