"""
$(TYPEDEF)

Interface for an optimal-design criterion.

Calling a criterion on an `AbstractMatrix` wraps it in `Symmetric`. Supply symmetric
information; criteria do not regularize it or check positive definiteness.
For a statistical covariance interpretation, positive-definite information is the
relevant case. Inverse criteria require invertibility. See [criteria](@ref oed_criteria)
for the exact definitions, which use neither logarithms nor dimension normalization.

Concrete criteria must implement `(::MyCriterion)(F::Symmetric)` and return a scalar
objective from a Fisher information matrix. The generic layer method calls
`fisher_information(layer, x, ps, st)` and returns `(value, st)` while preserving the
updated layer state. Criteria are pure with respect to the matrix input and must not
mutate the layer state.

# Example
```julia
using CorleoneOED, LinearAlgebra
struct TraceCriterion <: AbstractCriterion end
(::TraceCriterion)(F::Symmetric) = tr(F)
```
"""
abstract type AbstractCriterion end

"""
$(TYPEDEF)
Minimizes ``\\operatorname{tr}(F^{-1})``. This is an A-optimality criterion and is
defined for a nonsingular symmetric Fisher information matrix.
"""
struct ACriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Minimizes ``\\det(F^{-1})`` for a nonsingular symmetric Fisher information matrix.
"""
struct DCriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Minimizes the largest eigenvalue of ``F^{-1}`` for a symmetric Fisher information
matrix. It requires nonsingular information; repeated extreme eigenvalues can
make the objective nondifferentiable.
"""
struct ECriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Maximizes ``\\operatorname{tr}(F)`` by minimizing its negative.
"""
struct FisherACriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Maximizes ``\\det(F)`` by minimizing its negative.
"""
struct FisherDCriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Maximizes the smallest eigenvalue of `F` by minimizing its negative. Repeated
smallest eigenvalues can make the objective nondifferentiable.
"""
struct FisherECriterion <: AbstractCriterion end

function (crit::AbstractCriterion)(layer::Union{OEDLayer, MultiExperimentLayer}, x, ps, st::NamedTuple)
    F, st = fisher_information(layer, x, ps, st)
    return crit(F), st
end

function (crit::AbstractCriterion)(F::AbstractMatrix)
    return crit(Symmetric(F))
end

function (crit::ACriterion)(F::Symmetric)
    return tr(inv(F))
end

function (crit::DCriterion)(F::Symmetric)
    return inv(det(F))
end

function (crit::ECriterion)(F::Symmetric)
    return maximum(eigvals(inv(F)))
end

function (crit::FisherACriterion)(F::Symmetric)
    return -tr(F)
end

function (crit::FisherDCriterion)(F::Symmetric)
    return -det(F)
end

function (crit::FisherECriterion)(F::Symmetric)
    return -minimum(eigvals(F))
end
