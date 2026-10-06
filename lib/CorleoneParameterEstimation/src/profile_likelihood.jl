# --------------------------------------------------------------------------------------
# Profile-likelihood confidence intervals.
#
# The asymptotic (Jacobian-based) uncertainties reported by [`estimate_parameters`](@ref)
# are exact only in the linear least-squares limit; for genuinely nonlinear models they are
# a first-order approximation that can under- or over-cover, and they tell nothing about
# profile *shape* (an asymmetric likelihood, or a flat valley signalling non-identifiability).
#
# Profile likelihood addresses this by re-fitting the model with one parameter held fixed
# at a grid of values and asking where the (weighted) residual sum of squares crosses a
# likelihood-ratio threshold. The result is a [`ProfileLikelihoodEstimate`](@ref) carrying,
# for every fitted parameter, a [`ProfileCurve`](@ref) (parameter value vs. minimized RSS)
# and the confidence interval read off that curve. It is the gold-standard cross-check of
# the asymptotic intervals and directly exposes non-identifiability (a flat profile that
# never rises above the threshold gives an unbounded interval).
#
# Threshold. With Gaussian measurement noise the likelihood-ratio confidence set for a
# single parameter is { θ_i : 2 (log L_max - log L(θ_i)) ≤ χ²_{1-α}(1) }. Profiling over the
# residual scale σ in the unweighted case turns this into the classic (exactly-true in the
# linear limit, F-based) form used throughout nonlinear least-squares:
#
#     RSS(θ_i) ≤ RSS_min + σ̂² · t²_{1-α/2}(n-p),   σ̂² = RSS_min / (n-p)
#
# This is precisely the same σ̂² and Student-t critical value that drive the asymptotic t-CIs
# of [`estimate_parameters`](@ref), so in the linear/Gaussian limit a profile interval
# collapses onto the asymptotic one -- which is exactly what makes it a meaningful cross-check.
# For *weighted* residuals (all experiments prescribe a known σ) the residuals and J are the
# weighted ones and RSS_min is the weighted RSS; in that case the threshold uses the
# chi-square quantile χ²_{1-α}(1) instead of the t-scaled variance (matching the fact that
# the weighted covariance `(JᵀJ)⁻¹` carries no σ̂² factor).
# --------------------------------------------------------------------------------------

"""
    ProfileCurve

One parameter's profile-likelihood curve: the minimized (weighted) residual sum of squares
`rss` as a function of the parameter value `values`, with the parameter held fixed at each
grid point (all other parameters re-optimized). Used for plotting the profile and for
reading off the profile-likelihood confidence interval.

# Fields
$(FIELDS)
"""
struct ProfileCurve{V<:AbstractVector, T}
    "Index of the parameter in the shared model block `[u0; p]` this curve refers to"
    index::Int
    "Parameter name (or `nothing`)"
    name::Union{Nothing, AbstractString}
    "Grid of fixed values at which the profile was evaluated"
    values::V
    "Minimized (weighted) residual sum of squares at each grid value"
    rss::V
    "Minimum RSS over the curve (the global optimum `RSS_min`)"
    min_rss::T
    "RSS threshold below which a grid value lies inside the confidence interval"
    threshold::T
    "Lower endpoint of the profile confidence interval (or `-Inf` when not bracketed)"
    lower::T
    "Upper endpoint of the profile confidence interval (or `Inf` when not bracketed)"
    upper::T
end

"""
    ProfileLikelihoodEstimate

Result of a profile-likelihood confidence-interval calculation, from
[`profile_likelihood`](@ref). For every fitted parameter it stores a [`ProfileCurve`](@ref)
(the minimized RSS as a function of the parameter value) together with the confidence
interval read off that curve, plus the asymptotic interval (from
[`estimate_parameters`](@ref)) for direct cross-checking.

# Fields
$(FIELDS)
"""
struct ProfileLikelihoodEstimate{V, S, C, P<:AbstractVector, A}
    "Point estimates of the fitted (shared model) parameters `[u0; p]`"
    x::V
    "Parameter names (or `nothing`)"
    names::Union{Nothing, Vector{Symbol}}
    "Confidence level `1 - α` used for the intervals"
    confidence_level::S
    "Profile-likelihood confidence intervals as a vector of `(lower, upper)` tuples"
    confidence_intervals::C
    "One [`ProfileCurve`](@ref) per fitted parameter"
    profiles::P
    "The asymptotic (Jacobian-based) intervals, for cross-checking (`[u0; p]` block)"
    asymptotic_intervals::A
    "Number of fitted parameters `p` (shared model block)"
    n_parameters::Int
    "Number of observations `n`"
    n_observations::Int
    "Degrees of freedom `n - p` used for the residual-variance estimate"
    degrees_of_freedom::Int
end

"""
$(SIGNATURES)

Return the profile-likelihood confidence intervals (a vector of `(lower, upper)` tuples),
one per fitted parameter, from a [`ProfileLikelihoodEstimate`](@ref).
"""
confidence_interval(est::ProfileLikelihoodEstimate) = est.confidence_intervals

"""
$(SIGNATURES)

Return the vector of [`ProfileCurve`](@ref)s (one per fitted parameter) of a
[`ProfileLikelihoodEstimate`](@ref).
"""
profiles(est::ProfileLikelihoodEstimate) = est.profiles

# Re-assemble a *full* flat optimization vector with the value `θ` re-inserted at flat
# index `i` of a reduced vector `xred` (the `i`-th entry removed). `n` is the full length.
function _insert_at(xred::AbstractVector, i::Int, θ, n::Int)
    T = promote_type(eltype(xred), typeof(θ))
    xfull = Vector{T}(undef, n)
    j = 1
    for k in 1:n
        if k == i
            xfull[k] = θ
        else
            xfull[k] = xred[j]
            j += 1
        end
    end
    return xfull
end

# (Weighted) *observation* residual sum of squares at a full flat vector `xfull`. Only the
# observation residuals are summed -- multiple-shooting continuity residuals are consistency
# constraints with no statistical content and are excluded, exactly as in `estimate_parameters`.
function _observation_rss(problem::CorleoneParameterEstimationProblem, xfull::AbstractVector)
    layer = problem.layer
    n_obs = n_observations(get_data(layer))
    n_shoot = n_shooting_constraints(layer)
    r = similar(xfull, n_obs + n_shoot)
    problem.residual(r, xfull, nothing)
    return sum(abs2, view(r, 1:n_obs))
end

# Residual variance σ̂², degrees of freedom and Student-t / chi-square threshold used to
# turn a profile RSS into a confidence interval, mirroring `estimate_parameters`.
function _profile_threshold(problem::CorleoneParameterEstimationProblem, x_full::AbstractVector, α)
    layer = problem.layer
    n_obs = n_observations(get_data(layer))
    n_shoot = n_shooting_constraints(layer)
    p_total = length(x_full)
    dof = n_obs - p_total
    dof > 0 || throw(ArgumentError(
        "Need more observations than estimated parameters to estimate the noise level " *
        "(got $n_obs observation(s) for $p_total estimated parameters)."))
    rss = _observation_rss(problem, x_full)
    σ² = rss / dof
    weighted = all(e -> !isnothing(e.σ), experiments(get_data(layer)))
    if weighted
        # Known σ: residuals are weighted, `(JᵀJ)⁻¹` needs no σ̂² scaling; the likelihood-ratio
        # threshold is the chi-square quantile on the weighted RSS difference.
        tcrit2 = _chi_square_quantile(1, 1 - α)
    else
        # Unknown σ, profiled out: F-based threshold `σ̂² · t²_{1-α/2}(dof)`.
        tcrit2 = _t_quantile(dof, 1 - α / 2)^2
    end
    return rss, σ², dof, tcrit2
end

# Upper chi-square quantile (the `(1 - α)`-quantile of the `χ²_df` distribution) used to
# turn a profile RSS into a confidence interval. Delegated to `StatsFuns`, which provides an
# accurate special-function implementation (via the regularized incomplete gamma function)
# rather than a local hand-rolled approximation -- mirroring how `_t_quantile` is delegated
# to `StatsFuns.tdistinvcdf`.
function _chi_square_quantile(df::Int, q::Real)
    (0 <= q <= 1) || throw(ArgumentError("Quantile level must be in [0, 1] (got $q)."))
    return chisqinvcdf(df, q)
end

# Build a reduced `NonlinearLeastSquaresProblem` in which flat index `i` of the full
# optimization vector is held fixed at `θ` (so the remaining `n-1` entries are re-optimized).
# `problem` is a `CorleoneParameterEstimationProblem`; the returned problem reuses its
# residual (rebuilding the full vector behind the scenes) so any NonlinearSolve
# least-squares solver can be used.
function _profile_problem(
        problem::CorleoneParameterEstimationProblem, i::Int, θ, solver;
        x0red = nothing, kwargs...
    )
    xfull = vec(collect(problem.x0))
    n = length(xfull)
    mask = [k for k in 1:n if k != i]
    # Default to the raw initial guess; callers may pass `x0red` (e.g. the previous grid
    # point's solution) to warm-start the re-fit for smooth, robust continuation.
    x0 = isnothing(x0red) ? xfull[mask] : vec(collect(x0red))
    resid_full = problem.residual
    n_res = n_residuals(problem.layer)
    oop = (xred, p) -> begin
        xf = _insert_at(vec(xred), i, θ, n)
        r = similar(xred, n_res)
        resid_full(r, xf, p)
        r
    end
    if isnothing(problem.lb)
        lb = ub = nothing
    else
        lb = problem.lb[mask]
        ub = problem.ub[mask]
    end
    return NonlinearLeastSquaresProblem(
        NonlinearFunction{false}(oop), x0; lb = lb, ub = ub
    )
end

# Evaluate the profile at a single fixed value `θ` of flat index `i`: re-solve the reduced
# problem (warm-started from `x0red`, normally the previous grid point's solution, or the
# global optimum's free coordinates at the first point) and return the minimized
# observation RSS together with the recovered free variables, so the caller can warm-start
# the next grid point. On solver failure a non-finite RSS is returned together with the
# starting guess, so the value is treated as outside the confidence set and the
# continuation is not broken. Returns `(rss, xfree)`.
function _profile_rss_at(problem, i, θ, solver, x0red; kwargs...)
    n = length(problem.x0)
    # If fixing coordinate `i` leaves no free variables (single-parameter fit), there is
    # nothing to optimize: the profile RSS is just the RSS at the fully specified point.
    if n == 1
        xf = _insert_at(Float64[], i, θ, n)
        rss = try
            r = _observation_rss(problem, xf)
            isfinite(r) ? r : Inf
        catch
            Inf
        end
        return rss, Float64[]
    end
    redprob = _profile_problem(problem, i, θ, solver; x0red = x0red, kwargs...)
    sol = try
        solve(redprob, solver; kwargs...)
    catch
        return Inf, vec(collect(x0red))
    end
    xfree = SciMLBase.successful_retcode(sol) ? vec(collect(sol.u)) : vec(collect(x0red))
    rss = try
        xf = _insert_at(xfree, i, θ, n)
        r = _observation_rss(problem, xf)
        isfinite(r) ? r : Inf
    catch
        Inf
    end
    return rss, xfree
end

"""
    profile_likelihood(problem, sol; α = 0.05, names = nothing, solver = GaussNewton(),
                       n_points = 41, span = 6.0, expand = true, kwargs...)

Compute profile-likelihood confidence intervals for every fitted parameter of a solved
`CorleoneParameterEstimationProblem`.

For each fitted parameter the model is re-fit (with [`NonlinearSolve`](@ref)) holding that
parameter fixed at `n_points` values sweeping a range around its optimum, while the remaining
parameters (and, for multiple shooting, the local shooting states) are re-optimized. The
resulting profile of minimized (weighted) residual sum of squares is compared against the
likelihood-ratio threshold (see [`ProfileCurve`](@ref)); the crossings give the
`(1-α)`-level confidence interval. A profile that never rises above the threshold (a flat
valley) yields an unbounded interval, flagging that parameter as non-identifiable.

# Keywords
- `α`: Significance level (default `0.05` → 95% intervals).
- `names`: Optional vector of `Symbol`s naming the fitted parameters.
- `solver`: Least-squares solver used for the repeated re-fits (default `GaussNewton()`; use
  a bounded solver such as `BoundedLevenbergMarquardt` when the parameters have active bounds).
- `n_points`: Number of grid points per parameter (default `41`).
- `span`: Half width of the sweep as a multiple of the asymptotic standard error
  (default `6.0`). A per-parameter `AbstractVector` is also accepted.
- `expand`: Whether to widen the sweep if a crossing is not initially bracketed (default
  `true`), so off-centered profiles still yield finite intervals.
- `kwargs...`: Forwarded to the `solve` calls (tolerances, etc.).

# Returns
A [`ProfileLikelihoodEstimate`](@ref). Use [`confidence_interval`](@ref) for the profile
intervals and [`profiles`](@ref) for the per-parameter curves.
"""
function profile_likelihood(
        problem::CorleoneParameterEstimationProblem, sol;
        α = 0.05,
        names = nothing,
        solver = GaussNewton(),
        n_points::Int = 41,
        span = 6.0,
        expand::Bool = true,
        kwargs...,
    )
    layer = problem.layer
    x_full = collect(vec(sol.u))
    m = _n_shared(layer, problem.ps0s)
    x = collect(x_full[1:m])
    p = m
    n_obs = n_observations(get_data(layer))

    # Asymptotic estimate (for the grid span and for cross-checking the intervals).
    est0 = estimate_parameters(problem, sol; α = α, names = names)
    se = est0.std_errors
    asymp_ci = est0.confidence_intervals

    # Threshold for the profile RSS (also `σ̂² · t²` / `χ²`, see file header).
    rss_min, σ², dof, tcrit2 = _profile_threshold(problem, x_full, α)
    threshold = rss_min + σ² * tcrit2

    span = span isa AbstractVector ? collect(Float64, span) : fill(Float64(span), p)

    xnames = isnothing(names) ? nothing : collect(Symbol, names)
    profiles_i = Vector{ProfileCurve}(undef, p)
    cis = Vector{Tuple{Float64, Float64}}(undef, p)
    for i in 1:p
        se_i = se[i]
        base = (isfinite(se_i) && se_i > 0) ? se_i : abs(x[i]) * 0.1 + 1.0e-3
        lo = x[i] - span[i] * base
        hi = x[i] + span[i] * base
        # Free variables of the global optimum (all flat indices except `i`): warm start for
        # the bracket check and for the first grid point.
        xfree0 = vec(collect(x_full[[k for k in eachindex(x_full) if k != i]]))
        lo, hi = _ensure_bracket(problem, i, lo, hi, span[i] * base, threshold,
            solver, expand, xfree0; kwargs...)
        grid = range(lo, hi, length = n_points)
        vals = Vector{Float64}(undef, n_points)
        rss_vals = Vector{Float64}(undef, n_points)
        xwarm = vec(collect(xfree0))
        for (k, θ) in enumerate(grid)
            vals[k] = θ
            rss_vals[k], xwarm = _profile_rss_at(problem, i, θ, solver, xwarm; kwargs...)
        end
        lower, upper = _profile_ci_endpoints(vals, rss_vals, threshold, x[i])
        name = isnothing(xnames) ? nothing : string(xnames[i])
        profiles_i[i] = ProfileCurve(i, name, vals, rss_vals, rss_min, threshold, lower, upper)
        cis[i] = (lower, upper)
    end

    return ProfileLikelihoodEstimate(
        x, xnames, 1 - α, cis, profiles_i, asymp_ci, p, n_obs, dof
    )
end

"""
    profile_likelihood(pe::ParameterEstimationLayer, sol; kwargs...)

Convenience variant that constructs the [`CorleoneParameterEstimationProblem`](@ref) from
the layer `pe` (with default options) and calls [`profile_likelihood`](@ref).
"""
function profile_likelihood(pe::ParameterEstimationLayer, sol; kwargs...)
    prob = CorleoneParameterEstimationProblem(pe)
    return profile_likelihood(prob, sol; kwargs...)
end

# Widen the profiling bracket [lo, hi] until the likelihood-ratio threshold is crossed at
# both ends (so both CI endpoints are bracketed), or give up after a bounded number of
# doublings. Each doubling re-evaluates the profile only at the (new) extremes, so the cost
# stays small. `expand == false` keeps the user-supplied span unconditionally.
function _ensure_bracket(
        problem, i, lo, hi, halfwidth, threshold, solver, expand, xwarm; kwargs...
    )
    # Widen until the threshold is crossed at both ends (so both CI endpoints are
    # bracketed), with a bounded number of doublings. A profile that never rises above the
    # threshold even after this generous widening is treated as unbounded (flat valley ->
    # non-identifiable) by `_profile_ci_endpoints`. Each doubling evaluates two re-solves
    # at the new extremes (warm-started from the global-optimum free coordinates), so the
    # cost stays small.
    for _ in 1:8
        rlo, _ = _profile_rss_at(problem, i, lo, solver, xwarm; kwargs...)
        rhi, _ = _profile_rss_at(problem, i, hi, solver, xwarm; kwargs...)
        if ((!expand) || rlo > threshold) && (rhi > threshold)
            return lo, hi
        end
        expand || return lo, hi
        lo -= halfwidth
        hi += halfwidth
        halfwidth *= 2
    end
    return lo, hi
end

# Read the CI endpoints off a profile curve: the widest interval of grid values with
# `rss <= threshold` containing the optimum `x[i]`. The upper/lower crossing is linearly
# interpolated between the last inside and first outside grid point for accuracy. Returns
# `(-Inf, Inf)` when the profile never crosses the threshold (non-identifiable), and
# propagates ±Inf when an end is not bracketed.
function _profile_ci_endpoints(vals, rss, threshold, xopt)
    n = length(vals)
    inside = rss .<= threshold
    any(inside) || return -Inf, Inf
    imin = argmin(abs.(vals .- xopt))

    # Left edge: walk down from the optimum while points are still inside the CI. If we
    # reach the leftmost grid point and it is still inside, the profile never crossed the
    # threshold on this side (a flat valley -> non-identifiable) and the interval is
    # unbounded; otherwise interpolate the crossing between the last outside and first
    # inside point.
    ie = imin
    while ie > 1 && inside[ie - 1]
        ie -= 1
    end
    lower = if ie == 1 && inside[1]
        -Inf
    elseif ie == 1
        vals[1]
    else
        _interp_cross(vals[ie - 1], rss[ie - 1], vals[ie], rss[ie], threshold)
    end

    # Right edge: walk up from the optimum similarly.
    je = imin
    while je < n && inside[je + 1]
        je += 1
    end
    upper = if je == n && inside[n]
        Inf
    elseif je == n
        vals[n]
    else
        _interp_cross(vals[je], rss[je], vals[je + 1], rss[je + 1], threshold)
    end
    return lower, upper
end

# Linear interpolation of the crossing of `threshold` between two grid points `(x1, r1)`
# (outside, r1 > threshold) and `(x2, r2)` (inside, r2 <= threshold).
function _interp_cross(x1, r1, x2, r2, threshold)
    delta = r1 - r2
    delta == 0 && return (x1 + x2) / 2
    return x1 + (x2 - x1) * (r1 - threshold) / delta
end

# Printers --------------------------------------------------------------------------
function _print_profile_header(io::IO, est::ProfileLikelihoodEstimate)
    type_color, no_color = SciMLBase.get_colorizers(io)
    print(io, type_color, "ProfileLikelihoodEstimate", no_color, " with ")
    print(io, _count_label(est.n_parameters, "parameter"), " (")
    print(io, _count_label(est.n_observations, "observation"))
    return print(io, ", ", round(Int, 100 * est.confidence_level), "% CIs):\n")
end

function Base.show(io::IO, est::ProfileLikelihoodEstimate)
    _print_profile_header(io, est)
    for i in eachindex(est.x)
        name = isnothing(est.names) ? "x[$i]" : string(est.names[i])
        (lo, hi) = est.confidence_intervals[i]
        (alo, ahi) = est.asymptotic_intervals[i]
        println(
            io, "  ", name, " = ", est.x[i],
            "  profile CI: [", lo, ", ", hi, "]",
            "  (asymptotic: [", alo, ", ", ahi, "])"
        )
    end
    return nothing
end

function Base.show(io::IO, curve::ProfileCurve)
    name = isnothing(curve.name) ? "x[$(curve.index)]" : curve.name
    return print(
        io, "ProfileCurve(", name, ", ", length(curve.values), " points, ",
        "min RSS = ", curve.min_rss, ", CI = [", curve.lower, ", ", curve.upper, "])"
    )
end
