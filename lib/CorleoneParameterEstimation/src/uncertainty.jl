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
variance, ``\\mathrm{RSS}`` the sum of squared (weighted) *observation* residuals
(multiple-shooting continuity residuals are excluded), ``n`` the number of observations and
``p`` the number of fitted parameters (including any nuisance local shooting states). When the
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
    "Sum of squared (weighted) *observation* residuals at the optimum (multiple-shooting
    continuity residuals are excluded)"
    residual_norm::T
    "Degrees of freedom `n - p` (observation residuals minus estimated parameters)"
    degrees_of_freedom::Int
    "Confidence level used for `confidence_intervals`"
    confidence_level::T
    "Asymptotic confidence intervals as a vector of `(lower, upper)` tuples"
    confidence_intervals::C
    "Number of fitted parameters `p`"
    n_parameters::Int
    "Number of observations `n`"
    n_observations::Int
    "R² (coefficient of determination) of the weighted observation residuals"
    r_squared::T
    "Adjusted R² (penalized for the number of fitted parameters)"
    adjusted_r_squared::T
    "Parameter correlation matrix derived from `covariance`"
    correlation::M
    "Condition number of the reduced residual Jacobian (large values indicate near non-identifiability)"
    condition_number::T
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

The problem's [`AbstractParameterTransform`](@ref)s (if any) are applied automatically: the
reported `x`, `std_errors` and `confidence_intervals` are in *natural* units, the covariance
is propagated with the delta method and confidence intervals are formed by back-transforming
the internal endpoints (so e.g. a `Log`-transformed parameter gets a strictly positive,
asymmetric interval).
"""
function estimate_parameters(
        problem::CorleoneParameterEstimationProblem, sol; α = 0.05, names = nothing
    )
    layer = problem.layer
    x_full = vec(collect(sol.u))
    p_total = length(x_full)
    # Number of *model* parameters reported to the user. For a multiple-shooting layer the
    # flat vector also holds the local shooting initial conditions (nuisance variables), so
    # only the shared `[u0; p]` block is reported. `xφ` is in the internal (transformed)
    # space the optimizer works in; `x` are the corresponding natural parameter values.
    m = _n_shared(layer, problem.ps0s)
    transforms = problem.transforms
    xφ = collect(x_full[1:m])
    x = [from_internal(transforms[i], xφ[i]) for i in 1:m]
    p = m
    # Residual layout (see `residuals!`): all observation residuals first, then the
    # multiple-shooting continuity residuals. Only the *data* residuals carry statistical
    # information — the continuity residuals are consistency constraints that vanish at the
    # optimum — so RSS, σ̂² and the degrees of freedom are computed from the observation
    # residuals alone, and every estimated parameter (the shared model block *and* any
    # nuisance local shooting states) costs one degree of freedom of data.
    n_obs = n_observations(get_data(layer))
    n_shoot = problem.n_shooting_constraints
    n_obs + n_shoot >= p_total || throw(ArgumentError(
        "Need at least as many residuals as parameters to estimate uncertainties " *
        "(got $(n_obs) observation(s) + $(n_shoot) shooting constraint(s) for $(p_total) parameters)."
    ))
    r = collect(vec(sol.resid))
    rss = sum(abs2, view(r, 1:n_obs))
    dof = n_obs - p_total
    dof > 0 || throw(ArgumentError(
        "Need more observations than estimated parameters to estimate the noise level " *
        "(got $(n_obs) observation(s) for $(p_total) estimated parameters)."
    ))
    σ² = rss / dof

    # Jacobian of the (possibly weighted) residuals at the optimum. Allocate the residual
    # buffer with the element type of `x_full` (a `Dual` during ForwardDiff) so AD is dual-safe.
    J = _residual_jacobian(problem, x_full, n_obs + n_shoot)

    # Eliminate the local shooting states (nuisance variables) by profiling, then invert the
    # reduced normal matrix of the model block, `Jred = Jθ - Jλ (JλᵀJλ)⁻¹ Jλᵀ Jθ`. For a
    # single-shooting layer there are no local states and `Jred = J`.
    Jred = _profile_reduced_jacobian(J, m)

    # Invert the normal matrix. For *weighted* residuals (all experiments specify a known
    # `σ`) the covariance is `(JᵀJ)⁻¹` directly; the `σ̂²` scaling only applies when the
    # residuals are unweighted and the noise level has to be estimated from the residuals.
    weighted = all(e -> !isnothing(e.σ), experiments(get_data(layer)))
    Σφ = (weighted ? one(eltype(J)) : σ²) .* _safe_inverse(Jred' * Jred)

    # Transform the internal covariance to natural units with the delta method,
    # `Σ_θ = D Σ_φ Dᵀ` where `D = diag(dθ/dφ)`. The internal standard errors `seφ` are the
    # ones that are back-transformed to form the (generally asymmetric) confidence interval;
    # for `Identity`/`Affine` this is exactly the symmetric interval around `x`.
    d = [dθ_dφ(transforms[i], xφ[i]) for i in 1:m]
    D = Diagonal(d)
    Σ = D * Σφ * D'

    xnames = isnothing(names) ? nothing : collect(Symbol, names)
    seφ = sqrt.(max.(diag(Σφ), zero(eltype(Σφ))))
    se = sqrt.(max.(diag(Σ), zero(eltype(Σ))))
    # Two-sided Student-t quantile (positive upper-tail critical value).
    tcrit = _t_quantile(dof, 1 - α / 2)
    ci = [
        (
            from_internal(transforms[i], xφ[i] - tcrit * seφ[i]),
            from_internal(transforms[i], xφ[i] + tcrit * seφ[i]),
        ) for i in 1:m
    ]

    # Goodness-of-fit diagnostics. R² uses the *weighted* total sum of squares so that it is
    # consistent with the weighted RSS above.
    y_obs, w_obs = _observation_data(layer)
    r2 = _r_squared(rss, y_obs, w_obs)
    adj_r2 = isnan(r2) ? r2 : 1 - (1 - r2) * (n_obs - 1) / dof
    corr = _correlation_from_cov(Σ, se)
    # Condition number in the *natural* parameterization (`∂r/∂θ = Jred · diag(1/d)`), so
    # that the non-identifiability diagnostic is meaningful for the reported parameters.
    condJ = cond(Jred * Diagonal(inv.(d)))
    if condJ > 1.0e8
        @warn "The residual Jacobian is ill-conditioned (condition number $(condJ)); " *
              "some parameters may be non-identifiable. Treat the reported uncertainties " *
              "with caution." maxlog = 1
    end

    return ParameterEstimate(
        x, xnames, Σ, se, σ², rss, dof, 1 - α, ci, p, n_obs,
        r2, adj_r2, corr, condJ
    )
end

# Jacobian of the in-place residual `problem.residual` with respect to the full flat vector
# `x_full`. `ForwardDiff.jacobian` is not type-inferrable (inference sees a `Union` of a
# `Matrix` of the input element type and a `Matrix` of `Dual`s), so the concrete result type
# is asserted to keep `estimate_parameters` type-stable. The residual buffer is allocated
# with the element type of `x_full` (a `Dual` during ForwardDiff) so AD is dual-safe.
function _residual_jacobian(problem::CorleoneParameterEstimationProblem, x_full::AbstractVector{T}, n_res::Int) where {T}
    J = ForwardDiff.jacobian(x -> begin
        rr = similar(x, n_res)
        problem.residual(rr, x, nothing)
        rr
    end, x_full)
    return J::Matrix{T}
end

# For a single-shooting layer the whole Jacobian is kept. For multiple shooting, `m < ncol(J)`
# and the columns of the local shooting states are profiled out of the information matrix.
function _profile_reduced_jacobian(J::AbstractMatrix, m::Int)
    if m == size(J, 2)
        return J
    end
    Jθ = J[:, 1:m]
    Jλ = J[:, (m + 1):end]
    proj = Jλ * _safe_inverse(Jλ' * Jλ) * (Jλ' * Jθ)
    return Jθ - proj
end

# Invert a (symmetric positive-definite) normal/information matrix, falling back to a
# pseudo-inverse with a warning when it is singular or numerically rank-deficient (e.g. a
# non-identifiable parameter combination), so that a fit does not abort with a
# `SingularException` and the user is told the uncertainties are unreliable.
function _safe_inverse(A::AbstractMatrix)
    S = Matrix((A + A') / 2)
    F = cholesky(Symmetric(S); check = false)
    if issuccess(F)
        return inv(F)
    end
    @warn "The least-squares information matrix is singular or numerically rank-deficient; " *
          "parameter uncertainties may be unreliable (some parameters may be " *
          "non-identifiable)." maxlog = 1
    return pinv(S)
end

# Flattened (unweighted) observation values and their inverse noise weights in exactly the
# residual order used by `residuals!` (experiment by experiment; within a shared-grid
# experiment, column-major over `(n_obs, n_t)`; within a per-quantity experiment, quantity
# by quantity). Used to compute the weighted total sum of squares for R².
function _observation_data(pe::ParameterEstimationLayer)
    ys = Float64[]
    ws = Float64[]
    for exp in experiments(get_data(pe))
        if isnothing(exp.times)
            append!(ys, vec(exp.y))
            _append_weights!(ws, exp.σ, exp.y)
        else
            for k in eachindex(exp.y)
                append!(ys, exp.y[k])
                σk = exp.σ isa AbstractVector ? exp.σ[k] : exp.σ
                _append_weights!(ws, σk, exp.y[k])
            end
        end
    end
    return ys, ws
end

_append_weights!(ws, ::Nothing, y) = append!(ws, ones(length(y)))
_append_weights!(ws, σ::Number, y) = append!(ws, fill(1 / σ, length(y)))
function _append_weights!(ws, σ::AbstractMatrix, y)
    return append!(ws, vec(1 ./ σ))
end
function _append_weights!(ws, σ::AbstractVector, y)
    # Per-component noise shared across time points (shared-grid form): repeat each entry
    # per time column, matching the column-major flattening of `y`. For a per-quantity
    # series (`y` a vector) `size(y, 2) == 1`, so this is just the per-time weights.
    return append!(ws, repeat(1 ./ σ, size(y, 2)))
end

# Weighted coefficient of determination `R² = 1 - RSS / TSS`, with the weighted total sum of
# squares taken about the weighted mean of the observations. Returns `NaN` for a constant
# observation vector (undefined R²).
function _r_squared(rss, y, w)
    sw2 = sum(abs2, w)
    sw2 > 0 || return NaN
    ybar = sum(abs2.(w) .* y) / sw2
    tss = sum(abs2.(w) .* abs2.(y .- ybar))
    tss > 0 || return NaN
    return 1 - rss / tss
end

# Correlation matrix from a covariance matrix and the corresponding standard errors.
function _correlation_from_cov(Σ, se)
    C = similar(Σ, float(eltype(Σ)))
    for j in axes(Σ, 2), i in axes(Σ, 1)
        denom = se[i] * se[j]
        C[i, j] = denom > 0 ? Σ[i, j] / denom : (i == j ? one(eltype(C)) : zero(eltype(C)))
    end
    return C
end

"""
    estimate_parameters(pe::ParameterEstimationLayer, sol; α = 0.05, names = nothing)

Convenience variant that constructs the [`CorleoneParameterEstimationProblem`](@ref) from
`pe` (with default options) and calls [`estimate_parameters`](@ref). To control problem
construction (e.g. a custom `x0` or `shooting_scale`), build the problem explicitly first.

The `transforms` keyword attaches [`AbstractParameterTransform`](@ref)s to the shared `[u0; p]`
block; it must match the transforms the solution was computed with. When the problem is built
explicitly this keyword is not needed, since the problem already carries them.
"""
function estimate_parameters(
        pe::ParameterEstimationLayer, sol; α = 0.05, names = nothing, transforms = nothing
    )
    prob = CorleoneParameterEstimationProblem(pe; transforms = transforms)
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
    r_squared(est::ParameterEstimate)

Return the (weighted) coefficient of determination `R²` of the fit.
"""
r_squared(est::ParameterEstimate) = est.r_squared

"""
    adjusted_r_squared(est::ParameterEstimate)

Return the adjusted `R²` (penalized for the number of fitted parameters).
"""
adjusted_r_squared(est::ParameterEstimate) = est.adjusted_r_squared

"""
    correlation(est::ParameterEstimate)

Return the parameter correlation matrix derived from the estimated covariance.
"""
correlation(est::ParameterEstimate) = est.correlation

"""
    condition_number(est::ParameterEstimate)

Return the condition number of the reduced residual Jacobian at the optimum. Large values
indicate collinear (near non-identifiable) parameters and unreliable uncertainties.
"""
condition_number(est::ParameterEstimate) = est.condition_number

"""
    fitted_parameters(problem, sol; kwargs...)

Like [`estimate_parameters`](@ref), but returns a [`ParameterEstimationResult`](@ref) that
also carries the layer and the fitted trajectory. The `transforms` keyword behaves as for
[`estimate_parameters`](@ref).
"""
function fitted_parameters(problem::CorleoneParameterEstimationProblem, sol; kwargs...)
    est = estimate_parameters(problem, sol; kwargs...)
    traj = _fitted_trajectory(problem, sol)
    return ParameterEstimationResult(est, problem.layer, traj)
end

function fitted_parameters(pe::ParameterEstimationLayer, sol; transforms = nothing, kwargs...)
    prob = CorleoneParameterEstimationProblem(pe; transforms = transforms)
    return fitted_parameters(prob, sol; kwargs...)
end

function _fitted_trajectory(problem::CorleoneParameterEstimationProblem, sol)
    (; layer, ps0s, st, transforms) = problem
    x = collect(vec(sol.u))
    ps = _unflatten(layer, ps0s, x, transforms)
    traj, _ = layer(nothing, ps, st)
    return traj
end

# Two-sided Student-t quantile (positive upper-tail critical value), i.e. the inverse CDF of
# the Student-t distribution. Delegated to `StatsFuns`, which provides accurate special-function
# implementations (including the regularized incomplete beta function) rather than a local
# hand-rolled approximation.
function _t_quantile(ν::Int, q::Real)
    ν >= 1 || throw(ArgumentError(
        "The Student-t distribution needs at least 1 degree of freedom (got $ν)."))
    0 < q < 1 || throw(ArgumentError(
        "The quantile level must be in (0, 1) (got $q)."))
    return tdistinvcdf(ν, q)
end

# Shared parameter table used by the `ParameterEstimate` and `ParameterEstimationResult`
# printers: one line per parameter with its standard error and confidence interval.
function _print_parameter_table(io::IO, est::ParameterEstimate)
    pct = round(Int, 100 * est.confidence_level)
    for i in eachindex(est.x)
        name = isnothing(est.names) ? "x[$i]" : string(est.names[i])
        (lo, hi) = est.confidence_intervals[i]
        println(
            io, "  ", name, " = ", est.x[i], " ± ", est.std_errors[i],
            "  ($(pct)% CI: [", lo, ", ", hi, "])"
        )
    end
    return
end

# Header shared by both printers: "<Type> with p parameters (n observations):".
function _print_estimate_header(io::IO, name::AbstractString, est::ParameterEstimate)
    type_color, no_color = SciMLBase.get_colorizers(io)
    print(io, type_color, name, no_color, " with ")
    print(io, _count_label(est.n_parameters, "parameter"), " (")
    return print(io, _count_label(est.n_observations, "observation"), "):\n")
end

function Base.show(io::IO, est::ParameterEstimate)
    _print_estimate_header(io, "ParameterEstimate", est)
    _print_parameter_table(io, est)
    return print(io, "residual variance σ̂² = ", est.residual_variance)
end

# One-line description of the stored trajectory: `nothing` (not kept), a tuple of
# trajectories (one per experiment) or a single trajectory.
_trajectory_description(::Nothing) = "not stored"
_trajectory_description(trajectory::Tuple) = _count_label(length(trajectory), "experiment")
_trajectory_description(::Any) = "single experiment"

function Base.show(io::IO, result::ParameterEstimationResult)
    est = result.estimate
    _print_estimate_header(io, "ParameterEstimationResult", est)
    _print_parameter_table(io, est)
    print(io, "residual variance σ̂² = ", est.residual_variance)
    return print(io, "\nfitted trajectory: ", _trajectory_description(result.trajectory))
end
