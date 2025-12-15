abstract type AbstractCriterion end
"""
$(TYPEDEF)
Implements the ACriterion, i.e., ``\\textrm{tr}(F^{-1})\``.
"""
struct ACriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Implements the DCriterion, i.e., ``\\det(F^{-1})\``.
"""
struct DCriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Implements the ECriterion, i.e., ``\\max\\{\\lambda: \\lambda \\textrm{ is eigenvalue of } F^{-1}\\}\``.
"""
struct ECriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Implements the FisherACriterion, i.e., ``-\\textrm{tr}(F)\``
"""
struct FisherACriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Implements the FisherDCriterion, i.e., -``\\det(F)\``.
"""
struct FisherDCriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Implements the FisherECriterion, i.e., -``\\min\\{\\lambda: \\lambda \\textrm{ is eigenvalue of } F\\}\``.
"""
struct FisherECriterion <: AbstractCriterion end

function (crit::AbstractCriterion)(layer::OEDLayer, x, ps, st::NamedTuple)
	F,st = fisher_information(layer, x, ps, st)
	crit(F), st
end

function (crit::AbstractCriterion)(F::AbstractMatrix)
    crit(Symmetric(F))
end

function (crit::ACriterion)(F::Symmetric)
    tr(inv(F))
end

function (crit::DCriterion)(F::Symmetric)
    inv(det(F))
end

function (crit::ECriterion)(F::Symmetric)
    maximum(eigvals(inv(F)))
end

function (crit::FisherACriterion)(F::Symmetric)
    -tr(F)
end

function (crit::FisherDCriterion)(F::Symmetric)
    -det(F)
end

function (crit::FisherECriterion)(F::Symmetric)
    -minimum(eigvals(F))
end

