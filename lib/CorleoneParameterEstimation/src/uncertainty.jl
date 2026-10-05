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
    layer = problem.layer
    x_full = vec(collect(sol.u))
    p_total = length(x_full)
    # Number of *model* parameters reported to the user. For a multiple-shooting layer the
    # flat vector also holds the local shooting initial conditions (nuisance variables), so
    # only the shared `[u0; p]` block is reported.
    m = _n_shared(layer, problem.ps0s)
    x = collect(x_full[1:m])
    p = m
    # Residual layout (see `residuals!`): all observation residuals first, then the
    # multiple-shooting continuity residuals. Only the *data* residuals carry statistical
    # information — the continuity residuals are consistency constraints that vanish at the
    # optimum — so RSS, σ̂² and the degrees of freedom are computed from the observation
    # residuals alone, and every estimated parameter (the shared model block *and* any
    # nuisance local shooting states) costs one degree of freedom of data.
    n_obs = n_observations(get_data(layer))
    n_shoot = n_shooting_constraints(layer)
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
    J = ForwardDiff.jacobian(x -> begin
        rr = similar(x, n_obs + n_shoot)
        problem.residual(rr, x, nothing)
        rr
    end, x_full)

    # Eliminate the local shooting states (nuisance variables) by profiling, then invert the
    # reduced normal matrix of the model block, `Jred = Jθ - Jλ (JλᵀJλ)⁻¹ Jλᵀ Jθ`. For a
    # single-shooting layer there are no local states and `Jred = J`.
    Jred = _profile_reduced_jacobian(J, m)

    # Invert the normal matrix. For *weighted* residuals (all experiments specify a known
    # `σ`) the covariance is `(JᵀJ)⁻¹` directly; the `σ̂²` scaling only applies when the
    # residuals are unweighted and the noise level has to be estimated from the residuals.
    weighted = all(e -> !isnothing(e.σ), experiments(get_data(layer)))
    Σ = (weighted ? one(eltype(J)) : σ²) .* inv(Matrix(Jred' * Jred))

    xnames = isnothing(names) ? nothing : collect(Symbol, names)
    se = sqrt.(max.(diag(Σ), zero(eltype(Σ))))
    # Two-sided Student-t quantile (positive upper-tail critical value).
    tcrit = _t_quantile(dof, 1 - α / 2)
    ci = [(x[i] - tcrit * se[i], x[i] + tcrit * se[i]) for i in eachindex(x)]

    return ParameterEstimate(
        x, xnames, Σ, se, σ², rss, dof, 1 - α, ci, p, n_obs
    )
end

# For a single-shooting layer the whole Jacobian is kept. For multiple shooting, `m < ncol(J)`
# and the columns of the local shooting states are profiled out of the information matrix.
function _profile_reduced_jacobian(J::AbstractMatrix, m::Int)
    if m == size(J, 2)
        return J
    end
    Jθ = J[:, 1:m]
    Jλ = J[:, (m + 1):end]
    proj = Jλ * inv(Matrix(Jλ' * Jλ)) * (Jλ' * Jθ)
    return Jθ - proj
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
    (; layer, ps0s, st) = problem
    x = collect(vec(sol.u))
    ps = _unflatten(layer, ps0s, x)
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
