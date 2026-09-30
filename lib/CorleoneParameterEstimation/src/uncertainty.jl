"""
    ParameterEstimate

Point estimate of the fitted parameters together with their asymptotic uncertainties
obtained from the (weighted) least-squares covariance.

# Fields
$(FIELDS)

The covariance is estimated from the residual Jacobian ``J`` at the optimum,
```math
\\Sigma = \\hat{\\sigma}^2 \\, (J^\\top J)^{-1},
```
where ``\\hat{\\sigma}^2 = \\mathrm{RSS}/(n - p)`` is the unbiased estimate of the residual
variance, ``n`` the number of observations and ``p`` the number of fitted parameters. When the
data specifies measurement noise ``\\sigma``, the residuals (and hence ``J``) are the *weighted*
ones, so that ``(J^\\top J)^{-1}`` is the weighted least-squares covariance.
"""
struct ParameterEstimate{T, V, M, C}
    "Flat vector of fitted parameter values"
    x::V
    "Parameter names (or `nothing`)"
    names::Union{Nothing, Vector{Symbol}}
    "Estimated covariance matrix of the parameters"
    covariance::M
    "Estimated standard errors (`sqrt.(diag(covariance))`)"
    std_errors::V
    "Estimated residual variance `σ̂²`"
    residual_variance::T
    "Sum of squared (weighted) residuals at the optimum"
    residual_norm::T
    "Degrees of freedom `n - p`"
    degrees_of_freedom::Int
    "Confidence level used for `confidence_intervals`"
    confidence_level::T
    "Asymptotic confidence intervals as a vector of `(lower, upper)` tuples"
    confidence_intervals::C
    "Number of fitted parameters `p`"
    n_parameters::Int
    "Number of observations `n`"
    n_observations::Int
end

"""
    ParameterEstimationResult

Full result of a parameter-estimation run: the fitted parameter estimate together with the
corresponding layer (and, optionally, the fitted trajectory), so that the solution can be
plotted or re-evaluated.

# Fields
$(FIELDS)
"""
struct ParameterEstimationResult{E, L, S}
    "The parameter point estimate and uncertainties"
    estimate::E
    "The parameter-estimation layer"
    layer::L
    "The fitted trajectory (a single `Trajectory` for a single-experiment layer, or a tuple
    with one `Trajectory` per experiment), or `nothing`"
    trajectory::S
end

"""
    estimate_parameters(problem::CorleoneParameterEstimationProblem, sol; α = 0.05, names = nothing)

Compute the parameter estimates and their (asymptotic) uncertainties from the solved
`NonlinearLeastSquaresProblem` `problem` and its solution `sol`.

# Keywords
- `α`: Significance level for the confidence intervals (default `0.05` → 95% intervals).
- `names`: Optional vector of `Symbol`s naming the fitted parameters.

# Returns
A [`ParameterEstimate`](@ref).
"""
function estimate_parameters(
        problem::CorleoneParameterEstimationProblem, sol; α = 0.05, names = nothing
    )
    x = vec(collect(sol.u))
    p = length(x)
    n = n_observations(get_data(problem.layer))
    @assert n > p "Need more observations than parameters to estimate uncertainties (got $n observations, $p parameters)."

    r = collect(vec(sol.resid))
    rss = sum(abs2, r)
    dof = n - p
    σ² = rss / dof

    # Jacobian of the (possibly weighted) residuals at the optimum. Allocate the residual
    # buffer with the element type of `x` (a `Dual` during ForwardDiff) so AD is dual-safe.
    n_obs = n
    J = ForwardDiff.jacobian(x -> begin
        rr = similar(x, n_obs)
        problem.residual(rr, x, nothing)
        rr
    end, x)
    # Invert the normal matrix. For *weighted* residuals (all experiments specify a known
    # `σ`) the covariance is `(JᵀJ)⁻¹` directly; the `σ̂²` scaling only applies when the
    # residuals are unweighted and the noise level has to be estimated from the residuals.
    weighted = all(e -> !isnothing(e.σ), experiments(get_data(problem.layer)))
    Σ = (weighted ? one(eltype(J)) : σ²) .* inv(Matrix(J' * J))

    xnames = isnothing(names) ? nothing : collect(Symbol, names)
    se = sqrt.(max.(diag(Σ), zero(eltype(Σ))))
    # Two-sided asymptotic t-quantile (positive upper-tail critical value).
    tcrit = _t_quantile(dof, 1 - α / 2)
    ci = [(x[i] - tcrit * se[i], x[i] + tcrit * se[i]) for i in eachindex(x)]

    return ParameterEstimate(
        x, xnames, Σ, se, σ², rss, dof, 1 - α, ci, p, n
    )
end

"""
    estimate_parameters(pe::ParameterEstimationLayer, sol; α = 0.05, names = nothing)

Convenience variant that constructs the [`CorleoneParameterEstimationProblem`](@ref) from
`pe` (with default options) and calls [`estimate_parameters`](@ref). To control problem
construction (e.g. `include_ic`), build the problem explicitly first.
"""
function estimate_parameters(pe::ParameterEstimationLayer, sol; α = 0.05, names = nothing)
    prob = CorleoneParameterEstimationProblem(pe)
    return estimate_parameters(prob, sol; α = α, names = names)
end

"""
    covariance(est::ParameterEstimate)

Return the estimated covariance matrix of the parameters.
"""
covariance(est::ParameterEstimate) = est.covariance

"""
    standard_errors(est::ParameterEstimate)

Return the estimated standard errors of the parameters.
"""
standard_errors(est::ParameterEstimate) = est.std_errors

"""
    confidence_interval(est::ParameterEstimate)

Return the asymptotic confidence intervals as a vector of `(lower, upper)` tuples.
"""
confidence_interval(est::ParameterEstimate) = est.confidence_intervals

"""
    fitted_parameters(problem, sol; kwargs...)

Alias for [`estimate_parameters`](@ref) returning a [`ParameterEstimationResult`](@ref) that
also carries the layer and the fitted trajectory.
"""
function fitted_parameters(problem::CorleoneParameterEstimationProblem, sol; kwargs...)
    est = estimate_parameters(problem, sol; kwargs...)
    traj = _fitted_trajectory(problem, sol)
    return ParameterEstimationResult(est, problem.layer, traj)
end

fitted_parameters(pe::ParameterEstimationLayer, sol; kwargs...) =
    fitted_parameters(CorleoneParameterEstimationProblem(pe), sol; kwargs...)

function _fitted_trajectory(problem::CorleoneParameterEstimationProblem, sol)
    (; layer, ps0, st) = problem
    x = collect(vec(sol.u))
    ps = _unflatten(ps0, x)
    traj, _ = layer(nothing, ps, st)
    return traj
end

# Student-t quantile (asymptotic normal surrogate with small-sample correction via t).
function _t_quantile(dof::Int, q::Real)
    return _normal_quantile(q) * (1 + (1 + q * q) / (4 * dof))
end

function _normal_quantile(p::Real)
    # Rational approximation (Acklam) to the inverse normal CDF.
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00)
    plow = 0.02425
    phigh = 1 - plow
    q = p
    if q < plow
        r = sqrt(-2 * log(q))
        return (((((c[1] * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) * r + c[6]) /
               ((((d[1] * r + d[2]) * r + d[3]) * r + d[4]) * r + 1)
    elseif q <= phigh
        r = q - 0.5
        r2 = r * r
        return (((((a[1] * r2 + a[2]) * r2 + a[3]) * r2 + a[4]) * r2 + a[5]) * r2 + a[6]) * r /
               (((((b[1] * r2 + b[2]) * r2 + b[3]) * r2 + b[4]) * r2 + b[5]) * r2 + 1)
    else
        r = sqrt(-2 * log(1 - q))
        return -(((((c[1] * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5]) * r + c[6]) /
               ((((d[1] * r + d[2]) * r + d[3]) * r + d[4]) * r + 1)
    end
end

function Base.show(io::IO, est::ParameterEstimate)
    type_color, no_color = SciMLBase.get_colorizers(io)
    print(io, type_color, "ParameterEstimate", no_color, " with ")
    print(io, est.n_parameters, " parameters (", est.n_observations, " observations):\n")
    for i in eachindex(est.x)
        name = isnothing(est.names) ? "x[$i]" : string(est.names[i])
        (lo, hi) = est.confidence_intervals[i]
        println(
            io, "  ", name, " = ", est.x[i], " ± ", est.std_errors[i],
            "  (95%" * " CI: [", lo, ", ", hi, "])"
        )
    end
    return print(io, "residual variance σ̂² = ", est.residual_variance)
end
