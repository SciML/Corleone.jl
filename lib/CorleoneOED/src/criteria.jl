"""
$(TYPEDEF)

Interface for an optimal-design criterion.

Concrete criteria must implement `(::MyCriterion)(F::Symmetric)` and return a scalar
objective from a Fisher information matrix. The generic layer method calls
`fisher_information(layer, x, ps, st)` and returns `(value, st)` while preserving the
updated layer state. Criteria are pure with respect to the matrix input and must not
mutate the layer state.

# Example
```julia
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
matrix.
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
Maximizes the smallest eigenvalue of `F` by minimizing its negative.
"""
struct FisherECriterion <: AbstractCriterion end

function (crit::AbstractCriterion)(layer::OEDLayer, x, ps, st::NamedTuple)
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
