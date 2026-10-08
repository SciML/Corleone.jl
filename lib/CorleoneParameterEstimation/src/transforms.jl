# --------------------------------------------------------------------------------------
# Parameter transforms (reparameterizations)
#
# A transform is a bijection between the *natural* parameter ``\theta`` (the value that
# enters the model and is reported to the user) and the *internal* variable ``\phi`` that
# the optimizer actually works with:
#
#     \theta = from_internal(t, \phi),      \phi = to_internal(t, \theta).
#
# Fitting in ``\phi`` has three benefits over constraining ``\theta`` with bounds:
#
#   1. Constraints become part of the *geometry*: `Log` makes ``\theta > 0`` automatic,
#      `Logit` makes ``lb < \theta < ub`` automatic, and both leave the internal problem
#      *unconstrained*, so any unconstrained solver (and any AD backend) applies.
#   2. The internal problem is better scaled when parameters span orders of magnitude.
#   3. Uncertainties are propagated back to natural units correctly: the covariance is
#      transformed with the delta method, ``\Sigma_\theta = D \, \Sigma_\phi \, D^\top``
#      with ``D = \mathrm{diag}(d\theta/d\phi)``, and confidence intervals are formed by
#      back-transforming the *endpoints* ``\phi \pm t \, \mathrm{se}_\phi`` (so a
#      `Log`-space interval becomes correctly asymmetric and strictly positive).
# --------------------------------------------------------------------------------------

"""
    AbstractParameterTransform

Supertype of the parameter transforms accepted by [`CorleoneParameterEstimationProblem`](@ref)
through its `transforms` keyword. A transform is a smooth bijection between the *natural*
parameter value ``\\theta`` (used by the model and reported to the user) and the *internal*
value ``\\phi`` that the optimizer fits:

```math
\\theta = \\mathrm{from\\_internal}(t, \\phi), \\qquad \\phi = \\mathrm{to\\_internal}(t, \\theta).
```

# Implemented transforms
- [`Identity`](@ref): ``\\theta = \\phi`` (no transform, natural bounds are kept).
- [`Affine`](@ref): ``\\theta = a \\phi + b`` (scaling/centering, exactly linear).
- [`Log`](@ref): ``\\theta = e^{\\phi}`` (strict positivity, ``\\theta > 0``).
- [`Logit`](@ref): ``\\theta = lb + (ub - lb)\\,\\sigma(\\phi)`` (box constraint
  ``lb < \\theta < ub``).

Any transform other than `Identity` makes the corresponding internal variable unconstrained
(its natural bounds are dropped, since the transform's range encodes the constraint).
"""
abstract type AbstractParameterTransform end

"""
    Identity()

The trivial transform ``\\theta = \\phi``. It changes neither the parameterization nor the
bounds, and is the default for every optimization variable when no `transforms` are given.
"""
struct Identity <: AbstractParameterTransform end

"""
    Affine(a, b)

Linear (affine) transform ``\\theta = a\\phi + b`` with ``a \\neq 0``. Because the derivative
``d\\theta/d\\phi = a`` is constant, this is a pure rescaling/centering: it can be used to
bring parameters of very different magnitudes onto a common scale (improving the conditioning
of the least-squares problem) and it is the only non-trivial transform for which the delta
method and the back-transformed confidence interval are *exact* (the interval stays
symmetric).

# Example
```julia
# Fit a parameter that is naturally O(1e6) on an O(1) scale.
transforms = [Affine(1.0e6, 0.0)]
```
"""
struct Affine{T <: Real} <: AbstractParameterTransform
    "Scale `a` (`θ = a·φ + b`, must be non-zero)"
    a::T
    "Offset `b` (`θ = a·φ + b`)"
    b::T
    function Affine(a::Real, b::Real)
        iszero(a) && throw(ArgumentError(
            "Affine transform needs a non-zero scale `a` (got a = $a)."))
        return new{promote_type(typeof(a), typeof(b))}(promote(a, b)...)
    end
end

"""
    Log()

Logarithmic transform ``\\theta = e^{\\phi}`` (``\\phi = \\log \\theta``) enforcing strict
positivity ``\\theta > 0`` with an *unconstrained* internal variable. This is the natural
choice for rates, variances, scale parameters and other strictly positive quantities, and it
makes the back-transformed confidence interval asymmetric and automatically positive.
"""
struct Log <: AbstractParameterTransform end

"""
    Logit(lb, ub)

Logistic transform mapping the real line onto the open interval ``(lb, ub)``:
```math
\\theta = lb + (ub - lb)\\,\\sigma(\\phi), \\qquad \\sigma(\\phi) = \\frac{1}{1 + e^{-\\phi}}.
```
It enforces ``lb < \\theta < ub`` while leaving the internal variable ``\\phi`` unconstrained,
so a box-constrained fit can be solved with an *unbounded* solver (and hence with any AD
backend, avoiding the mutating bounds transformation of bounded solvers).

# Example
```julia
# Keep a rate in (0, 1) without using a bounded solver.
transforms = [Logit(0.0, 1.0)]
```
"""
struct Logit{T <: Real} <: AbstractParameterTransform
    "Lower bound of the interval (excluded)"
    lb::T
    "Upper bound of the interval (excluded)"
    ub::T
    function Logit(lb::Real, ub::Real)
        lb < ub || throw(ArgumentError(
            "Logit transform needs `lb < ub` (got lb = $lb, ub = $ub)."))
        return new{promote_type(typeof(lb), typeof(ub))}(promote(lb, ub)...)
    end
end

# --- Transform algebra --------------------------------------------------------------

"""
    to_internal(t::AbstractParameterTransform, θ)

Map a natural parameter value `θ` to the internal variable `φ` fitted by the optimizer
(the inverse of [`from_internal`](@ref)).
"""
to_internal(::Identity, θ) = θ
to_internal(t::Affine, θ) = (θ - t.b) / t.a
to_internal(::Log, θ) = log(θ)
function to_internal(t::Logit, θ)
    p = (θ - t.lb) / (t.ub - t.lb)
    return log(p) - log1p(-p)
end

"""
    from_internal(t::AbstractParameterTransform, φ)

Map the internal variable `φ` fitted by the optimizer back to the natural parameter value
`θ` used by the model (the inverse of [`to_internal`](@ref)).
"""
from_internal(::Identity, φ) = φ
from_internal(t::Affine, φ) = t.a * φ + t.b
from_internal(::Log, φ) = exp(φ)
from_internal(t::Logit, φ) = t.lb + (t.ub - t.lb) * _logistic(φ)

"""
    dθ_dφ(t::AbstractParameterTransform, φ)

Derivative ``d\\theta/d\\phi`` of the natural parameter with respect to the internal variable
at `φ`. Used for the delta-method covariance transformation and for back-transforming
confidence-interval endpoints.
"""
dθ_dφ(::Identity, φ) = one(φ)
dθ_dφ(t::Affine, φ) = t.a
dθ_dφ(::Log, φ) = exp(φ)
function dθ_dφ(t::Logit, φ)
    s = _logistic(φ)
    return (t.ub - t.lb) * s * (one(s) - s)
end

# Numerically stable logistic function. The naive `1/(1 + exp(-φ))` already saturates
# correctly: for large negative `φ`, `exp(-φ)` overflows to `Inf` and the result is `0`.
_logistic(φ) = inv(one(φ) + exp(-φ))

# Internal bounds implied by a transform. `Identity` keeps the natural bounds; every other
# transform is a bijection onto its (possibly constrained) range and therefore needs no
# internal bounds.
_internal_bounds(t::Identity, lb, ub) = (lb, ub)
_internal_bounds(::AbstractParameterTransform, lb, ub) = (-Inf, Inf)
