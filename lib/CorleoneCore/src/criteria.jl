
abstract type AbstractOEDCriterion end

struct ACriterion{T} <: AbstractOEDCriterion where T <: Tuple
    tspan::T
    function ACriterion(tspan)
        return new{typeof(tspan)}(tspan)
    end
end

struct DCriterion{T} <: AbstractOEDCriterion
    tspan::T
    function DCriterion(tspan)
        return new{typeof(tspan)}(tspan)
    end
end

struct ECriterion{T} <: AbstractOEDCriterion
    tspan::T
    function ECriterion(tspan)
        return new{typeof(tspan)}(tspan)
    end
end


function _symmetric_from_vector(x::AbstractArray{T}, ::Val{N}, regu) where {T, N}
    F = Array{T,2}(undef,N,N)
    for j=1:N
        for i=1:N
            if i<= j
                F[i,j] = x[Int(j * (j - 1) / 2 + i)]
                if i==j
                    F[i,j] += regu
                end
            else
                F[i,j] = x[Int(i * (i - 1) / 2 + j)]
            end
        end
    end
    return F

    #return Symmetric([i <= j ? x[Int(j * (j - 1) / 2 + i)] : zero(T) for i in 1:N, j in 1:N])
end

function __symmetric_from_vector(x::AbstractVector, regu)
    n = Int(sqrt(2 * size(x, 1) + 0.25) - 0.5)
    _symmetric_from_vector(x, Val(n), regu)
end

#function __symmetric_from_vector(x::AbstractVector)
#    n = Int(sqrt(2 * size(x, 1) + 0.25) - 0.5)
#    _symmetric_from_vector(x, Val(n), regu)
#end


#function (crit::ACriterion)(F::AbstractVector)
#    return crit(__symmetric_from_vector(F))
#end
#
#function (crit::DCriterion)(F::AbstractVector)
#    return crit(__symmetric_from_vector(F))
#end
#
#function (crit::ECriterion)(F::AbstractVector)
#    return crit(__symmetric_from_vector(F))
#end

function (crit::ACriterion)(F::AbstractMatrix)
    return inv(tr(F))
end

function (crit::DCriterion)(F::AbstractMatrix)
    return inv(det(F))
end

function (crit::ECriterion)(F::AbstractMatrix)
    return max(eigvals(F))
end