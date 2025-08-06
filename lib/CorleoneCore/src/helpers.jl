# Piecewise constant control function
function find_index(timestops, t)
    idx = searchsortedlast(timestops, t)
    # We always assume that we have left / right continuity
    min(max(firstindex(timestops), idx), lastindex(timestops))
end

function δₜ(p, timestops, t)
    @assert size(timestops, 1) == size(p, 1) "The dimensionality of the provided `tstops` and `parameters` are not consistent."
    id = find_index(timestops, t)
    getindex(p, id)
end

function replace_portion(x::AbstractArray{X}, y::AbstractArray{Y}, indices::AbstractVector) where {X, Y}
    T = Base.promote_type(X, Y)
    isempty(indices) && return T.(x)
    xreplace = [i ∈ b for i in eachindex(x)]
    all(xreplace) && return T.(y)
    
    #return xkeep .* T.(x) .+ xreplace .* T.(y)
end

