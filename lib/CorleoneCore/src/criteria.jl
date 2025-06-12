
abstract type AbstractOEDCriterion end

struct ACriterion <: AbstractOEDCriterion end
struct DCriterion <: AbstractOEDCriterion end
struct ECriterion <: AbstractOEDCriterion end



function _symmetric_from_vector(x::AbstractArray{T}, ::Val{N}) where {T, N}
    return Symmetric([i <= j ? x[Int(j * (j - 1) / 2 + i)] : zero(T) for i in 1:N, j in 1:N])
end

function __symmetric_from_vector(x::AbstractArray)
    n = Int(sqrt(2 * size(x, 1) + 0.25) - 0.5)
    _symmetric_from_vector(x, Val(n))
end

function (crit::ACriterion)(F::AbstractArray)
    return crit(__symmetric_from_vector(F))
end

function (crit::DCriterion)(F::AbstractArray)
    return crit(__symmetric_from_vector(F))
end

function (crit::ECriterion)(F::AbstractArray)
    return crit(__symmetric_from_vector(F))
end

function (crit::ACriterion)(F::AbstractMatrix)
    return inv(tr(F))
end

function (crit::DCriterion)(F::AbstractMatrix)
    return inv(det(F))
end

function (crit::ECriterion)(F::AbstractMatrix)
    return max(eigvals(F))
end