abstract type AbstractCriterion end

struct ACriterion <: AbstractCriterion end
struct DCriterion <: AbstractCriterion end
struct ECriterion <: AbstractCriterion end

struct FisherACriterion <: AbstractCriterion end
struct FisherDCriterion <: AbstractCriterion end
struct FisherECriterion <: AbstractCriterion end

function (crit::ACriterion)(F::Symmetric)
    tr(inv(F))
end

function (crit::DCriterion)(F::Symmetric)
    det(inv(F))
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

(crit::AbstractCriterion)(layer::SingleShootingLayer, sols::DiffEqArray) = begin
    f_sym = CorleoneCore.fisher_variables(layer)
    crit((CorleoneCore.symmetric_from_vector(sols[f_sym][end])))
end

(crit::AbstractCriterion)(layer::MultipleShootingLayer, sols::EnsembleSolution) = begin
    f_sym = CorleoneCore.fisher_variables(layer)
    crit(CorleoneCore.symmetric_from_vector(last(sols)[f_sym][end]))
end
