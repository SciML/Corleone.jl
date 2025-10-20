abstract type AbstractCriterion end
"""
$(TYPEDEF)
Implements the ACriterion, e.g., tr(F^-1))
"""
struct ACriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Implements the DCriterion, e.g., det(F^-1))
"""
struct DCriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Implements the ECriterion, e.g., max{λ: λ is eigenvalue of F^-1}
"""
struct ECriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Implements the FisherACriterion, e.g., -tr(F))
"""
struct FisherACriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Implements the FisherDCriterion, e.g., -det(F))
"""
struct FisherDCriterion <: AbstractCriterion end
"""
$(TYPEDEF)
Implements the FisherECriterion, e.g., -min{λ: λ is eigenvalue of F}
"""
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

function fim(layer::SingleShootingLayer, sols::DiffEqArray)
    f_sym = Corleone.fisher_variables(layer)
    Corleone.symmetric_from_vector(sols[f_sym][end])
end

function fim(layer::MultipleShootingLayer, sols::EnsembleSolution)
    f_sym = Corleone.fisher_variables(layer)
    Corleone.symmetric_from_vector(last(sols)[f_sym][end])
end

(crit::AbstractCriterion)(layer::SingleShootingLayer, sols::DiffEqArray) = begin
    crit(fim(layer, sols))
end

(crit::AbstractCriterion)(layer::MultipleShootingLayer, sols::EnsembleSolution) = begin
    crit(fim(layer, sols))
end
