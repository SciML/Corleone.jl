struct NoShoot <: AbstractAutoShoot end 

function apply_auto_shoot(::NoShoot, args...) 
    return []
end 

"""
$(TYPEDEF)

Injects fixed shooting points into the corresponding layers.
"""
struct FixedShoot{T} <: AbstractAutoShoot
    "Shooting points"
    tpoints::T
end 

function apply_auto_shoot(method::FixedShoot, args...)
    deepcopy(collect(method.tpoints))
end

"""
$(TYPEDEF)

Finds the best timepoints to inject so that the resulting structure will contain exactly `n` blocks.
"""
struct AutoBlock <: AbstractAutoShoot
    "Number of target blocks"
    n::Int64 
end

function apply_auto_shoot(method::AutoBlock, activity_patterns, timepoints)
    (; n) = method
    # Parameter conts 
    A = reduce(hcat, activity_patterns.controls)
    S = size(A, 1)
    w = [count(j -> A[s, j] & A[s+1,j], axes(A, 2)) for s in 1:S-1]
    best_options = partialsortperm(w, Base.OneTo(n))
    timepoints[best_options]
end