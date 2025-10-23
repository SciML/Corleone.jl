abstract type AbstractCriterion end

struct ACriterion <: AbstractCriterion end
struct DCriterion <: AbstractCriterion end
struct ECriterion <: AbstractCriterion end

struct FisherACriterion <: AbstractCriterion end
struct FisherDCriterion <: AbstractCriterion end
struct FisherECriterion <: AbstractCriterion end

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
