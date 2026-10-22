# --------------------------------------------------------------------------------------
# Parameter regularization (ridge / lasso / elastic-net / custom).
#
# Regularization adds a data-independent penalty to the least-squares objective to favour
# small (or smooth) parameters and to stabilize ill-conditioned or overparameterized fits.
# Following the package's conventions it is expressed as *additional residuals* appended to
# the residual vector after the observation residuals and the multiple-shooting continuity
# residuals, so that the least-squares objective becomes
#
#     ½ ‖r‖² = ½·RSS_data + ½·‖r_reg‖²
#            = ½·RSS_data + λ·Σ wᵢ² (θᵢ - refᵢ)²        (L2 / ridge)
#            = ½·RSS_data + λ·Σ wᵢ |θᵢ - refᵢ|          (L1 / lasso)
#
# where `{θᵢ}` are the *penalized* model parameters (by default the model parameters `p`
# only, never the tuned initial conditions `u0` nor the local multiple-shooting states).
#
# The penalty is evaluated on the **natural** parameter values ``\theta`` (the values that
# enter the model and are reported to the user), recovered from the internal fitted
# variable ``\phi`` by the problem's `transforms`. `reference` and `weights` are therefore
# expressed in natural units.
#
# Because the penalty enters as extra residuals, it is completely transparent to the
# solvers: NonlinearSolve's Gauss-Newton / Levenberg-Marquardt / trust-region solvers and the
# Optimization.jl extension (whose scalar objective is `0.5 * sum(abs2, r)`) all minimize the
# penalized objective without any solver-specific code. The appended penalty residuals also
# enter the residual Jacobian used by `estimate_parameters`, so the reported covariance is
# the MAP / posterior covariance `(J_dataᵀJ_data + Λ)⁻¹·σ̂²` (see `uncertainty.jl`).
# --------------------------------------------------------------------------------------

"""
    AbstractRegularizer

Supertype of the parameter regularizers accepted by
[`CorleoneParameterEstimationProblem`](@ref) through its `regularization` keyword.

A regularizer augments the least-squares objective with a data-independent penalty on the
fitted *model parameters* (by default the model parameters `p`, never the tuned initial
conditions or the local multiple-shooting states). A regularizer is a callable

```julia
(r::AbstractRegularizer)(θ_pen) -> r_reg::AbstractVector
```

mapping the vector of penalized **natural** parameter values `θ_pen` to a vector of
additional residual contributions `r_reg`. Appending these residuals leaves the objective
as ``\\tfrac12\\|r_{\\mathrm{data}}\\|^2 + \\tfrac12\\|r_{\\mathrm{reg}}\\|^2``.

# Built-in regularizers
- [`L2Regularizer`](@ref): ``\\lambda \\sum_i w_i^2 (\\theta_i - \\mathrm{ref}_i)^2`` (ridge).
- [`L1Regularizer`](@ref): ``\\lambda \\sum_i w_i |\\theta_i - \\mathrm{ref}_i|`` (lasso).
- [`ElasticNetRegularizer`](@ref): combination of `L2` and `L1`.

Any callable matching the signature above can be used directly (e.g. a closure or a struct
returning custom residual contributions), so new regularization terms can be added without
changing the package.
"""
abstract type AbstractRegularizer end

# --- Residual count and residuals -----------------------------------------------------
# Every regularizer must report how many residual rows it contributes for `n` penalized
# parameters. This is resolved once at `CorleoneParameterEstimationProblem` construction so
# that `n_regularization_residuals` is a concrete `Int` field.

"""
    n_regularization_residuals(r::AbstractRegularizer, n::Int)

Number of residual rows contributed by regularizer `r` for `n` penalized parameters.
"""
n_regularization_residuals(::AbstractRegularizer, n::Int) = n

# Evaluate one regularizer on a vector of penalized natural parameters, returning a new
# residual vector. Every concrete regularizer must implement this; custom callables are
# invoked directly.
(reg::AbstractRegularizer)(θ_pen::AbstractVector) = reg(θ_pen)

"""
    L2Regularizer(λ; reference = nothing, weights = nothing)

Ridge (``L_2``) regularization penalizing the squared deviation of the penalized model
parameters from a reference value,

```math
\\tfrac12\\|r_{\\mathrm{reg}}\\|^2 = \\lambda \\sum_i w_i^2 \\,(\\theta_i - \\mathrm{ref}_i)^2 .
```

The penalty residuals are ``\\sqrt{\\lambda}\\,w_i\\,(\\theta_i - \\mathrm{ref}_i)``, so the
strength `λ` is measured relative to the (measurement-noise-weighted) observation residuals:
``\\lambda = 1`` adds residual rows comparable to a single ``O(1)`` observation residual.

# Arguments
- `λ`: Regularization strength (must be `> 0`).

# Keywords
- `reference`: Vector of shrinkage targets (or `nothing` for zeros), one per penalized
  parameter, in **natural** units.
- `weights`: Per-parameter scaling (or `nothing` for ones), one per penalized parameter.

`reference`/`weights` may also be supplied through the `regularization_reference` /
`regularization_weights` keywords of [`CorleoneParameterEstimationProblem`](@ref), which
take precedence.
"""
struct L2Regularizer{T, Vv <: AbstractVector} <: AbstractRegularizer
    λ::T
    reference::Vv
    weights::Vv
    function L2Regularizer(λ::Real; reference = nothing, weights = nothing)
        λ > 0 || throw(ArgumentError("L2Regularizer needs λ > 0 (got $λ)."))
        return new{promote_type(typeof(λ), Float64),
            Vector{promote_type(typeof(λ), Float64)}}(
            promote_type(typeof(λ), Float64)(λ),
            isnothing(reference) ? Float64[] : collect(Float64, reference),
            isnothing(weights) ? Float64[] : collect(Float64, weights),
        )
    end
end

function (reg::L2Regularizer)(θ_pen::AbstractVector)
    ref = isempty(reg.reference) ? zero.(θ_pen) : reg.reference
    w = isempty(reg.weights) ? one.(θ_pen) : reg.weights
    return sqrt(reg.λ) .* w .* (θ_pen .- ref)
end

"""
    L1Regularizer(λ; reference = nothing, weights = nothing)

Lasso (``L_1``) regularization penalizing the absolute deviation of the penalized model
parameters from a reference value,

```math
\\tfrac12\\|r_{\\mathrm{reg}}\\|^2 = \\lambda \\sum_i w_i \\,|\\theta_i - \\mathrm{ref}_i| .
```

The penalty residuals are ``\\sqrt{\\lambda}\\,\\sqrt{w_i\\,|\\theta_i - \\mathrm{ref}_i|}``
kept as separate rows per parameter (see [`n_regularization_residuals`](@ref)); equivalently
the penalty contributes ``\\lambda \\sum_i w_i |\\theta_i - \\mathrm{ref}_i|`` to ``\\tfrac12\\|r\\|^2``.
This is **non-smooth** at ``\\theta_i = \\mathrm{ref}_i``: Gauss-Newton-type solvers treat it
through its (biased at the kink) derivative, which is generally fine for pulling parameters
toward the reference but does *not* reliably produce exact zeros. For smooth,
well-conditioned shrinkage prefer [`L2Regularizer`](@ref) or [`ElasticNetRegularizer`](@ref).

# Arguments
- `λ`: Regularization strength (must be `> 0`).

# Keywords
- `reference`: Vector of shrinkage targets (or `nothing` for zeros), one per penalized
  parameter, in **natural** units.
- `weights`: Per-parameter scaling (or `nothing` for ones), one per penalized parameter.
"""
struct L1Regularizer{T, Vv <: AbstractVector} <: AbstractRegularizer
    λ::T
    reference::Vv
    weights::Vv
    function L1Regularizer(λ::Real; reference = nothing, weights = nothing)
        λ > 0 || throw(ArgumentError("L1Regularizer needs λ > 0 (got $λ)."))
        return new{promote_type(typeof(λ), Float64),
            Vector{promote_type(typeof(λ), Float64)}}(
            promote_type(typeof(λ), Float64)(λ),
            isnothing(reference) ? Float64[] : collect(Float64, reference),
            isnothing(weights) ? Float64[] : collect(Float64, weights),
        )
    end
end

function (reg::L1Regularizer)(θ_pen::AbstractVector)
    ref = isempty(reg.reference) ? zero.(θ_pen) : reg.reference
    w = isempty(reg.weights) ? one.(θ_pen) : reg.weights
    δ = θ_pen .- ref
    # sqrt of the per-row penalty so that ½ ‖r_reg‖² = λ Σ wᵢ |δᵢ|.
    return sqrt(reg.λ) .* sqrt.(w .* abs.(δ))
end

"""
    ElasticNetRegularizer(λ1, λ2; reference = nothing, weights = nothing)

Elastic-net regularization combining an [`L1Regularizer`](@ref) of strength `λ1` with an
[`L2Regularizer`](@ref) of strength `λ2`:

```math
\\tfrac12\\|r_{\\mathrm{reg}}\\|^2 = \\lambda_1 \\sum_i w_i |\\theta_i - \\mathrm{ref}_i|
    + \\lambda_2 \\sum_i w_i^2 (\\theta_i - \\mathrm{ref}_i)^2 .
```

# Arguments
- `λ1`, `λ2`: L1 and L2 strengths (both must be `> 0`).
# Keywords
- `reference`, `weights`: as for [`L2Regularizer`](@ref) / [`L1Regularizer`](@ref).
"""
struct ElasticNetRegularizer{T, Vv <: AbstractVector} <: AbstractRegularizer
    l1::L1Regularizer{T, Vv}
    l2::L2Regularizer{T, Vv}
    function ElasticNetRegularizer(λ1::Real, λ2::Real; reference = nothing, weights = nothing)
        return new{promote_type(typeof(λ1), typeof(λ2), Float64),
            Vector{promote_type(typeof(λ1), typeof(λ2), Float64)}}(
            L1Regularizer(λ1; reference = reference, weights = weights),
            L2Regularizer(λ2; reference = reference, weights = weights),
        )
    end
end

n_regularization_residuals(reg::ElasticNetRegularizer, n::Int) =
    n_regularization_residuals(reg.l1, n) + n_regularization_residuals(reg.l2, n)

function (reg::ElasticNetRegularizer)(θ_pen::AbstractVector)
    return vcat(reg.l1(θ_pen), reg.l2(θ_pen))
end

# --------------------------------------------------------------------------------------
# Resolution helpers used by the `CorleoneParameterEstimationProblem` constructor.
# --------------------------------------------------------------------------------------

# Resolve the penalized shared-block indices `indices` (1-based entries of the shared
# `[u0; p]` block). `nothing` selects the model-parameter block `(nu+1):m` (the default,
# so tuned initial conditions `u0` are *never* regularized unless explicitly requested); an
# explicit vector may select initial conditions, parameters, or any subset of the shared
# block. Local multiple-shooting states (beyond the shared block) are never available here.
function _resolve_penalized_indices(indices, nu::Int, np::Int)
    m = nu + np
    if isnothing(indices)
        return collect((nu + 1):m)
    end
    idx = collect(Int, indices)
    all(i -> 1 <= i <= m, idx) || throw(ArgumentError(
        "`regularized_indices` may only contain entries of the shared `[u0; p]` block " *
        "(1:$m); got $idx."))
    return idx
end

# Resolve the user-facing `regularization` keyword and the set of penalized shared-block
# entries. Returns `(reg, pen_indices, n_reg)`:
#   - `reg`: the resolved regularizer (a callable `reg(θ_pen) -> r_reg`), or `nothing` when
#     no regularization is in effect.
#   - `pen_indices`: the shared-block indices that are penalized; the natural penalized
#     parameters are `[from_internal(transforms[i], x[i]) for i in pen_indices]`.
#   - `n_reg`: the number of regularization residual rows.
# The `reference` / `weights` carried by the regularizer (or supplied via
# `regularization_reference` / `regularization_weights`) are aligned with `pen_indices`.
function _resolve_regularization(regularization, indices, nu::Int, np::Int)
    if isnothing(regularization)
        return (nothing, Int[], 0)
    end
    pen = _resolve_penalized_indices(indices, nu, np)
    n_pen = length(pen)
    _check_reg_vectors(regularization, n_pen)
    nreg = _regularizer_length(regularization, n_pen)
    return (regularization, pen, nreg)
end

# Validate that any non-empty `reference` / `weights` the user supplied to a built-in
# regularizer (or via the `regularization_reference` / `regularization_weights` overrides)
# match the number of penalized entries. The default (empty) vectors mean "zeros" /
# "ones" respectively and need no check.
function _check_reg_vectors(reg::AbstractRegularizer, n_pen::Int)
    for (name, v) in ((:reference, reg.reference), (:weights, reg.weights))
        if !isempty(v) && length(v) != n_pen
            throw(ArgumentError(
                "A regularizer `$name` vector has length $(length(v)) but $n_pen entry(ies) " *
                "are penalized; provide one value per penalized shared-block entry."))
        end
    end
    return nothing
end
function _check_reg_vectors(reg::ElasticNetRegularizer, n_pen::Int)
    _check_reg_vectors(reg.l1, n_pen)
    _check_reg_vectors(reg.l2, n_pen)
    return nothing
end
# Custom functors carry arbitrary state -> no length check.
_check_reg_vectors(::Any, n_pen::Int) = nothing

# Number of residual rows contributed by a regularizer for `np` penalized parameters. The
# built-in `AbstractRegularizer`s implement `n_regularization_residuals`; a plain functor
# (closure/struct) is probed once on a zero vector (finite and data-independent) so that
# its residual length can be fixed at construction time.
_regularizer_length(reg::AbstractRegularizer, np::Int) = n_regularization_residuals(reg, np)
function _regularizer_length(reg, np::Int)
    r = try
        reg(zeros(np))
    catch
        throw(ArgumentError(
            "Could not determine the number of regularization residuals from the custom " *
            "regularizer. Return a vector for a zero input, or define " *
            "`CorleoneParameterEstimation.n_regularization_residuals(reg, n)`."))
    end
    r isa AbstractVector || throw(ArgumentError(
        "A custom regularizer must return a vector of residuals (got $(typeof(r)))."))
    return length(r)
end
