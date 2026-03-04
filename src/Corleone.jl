module Corleone

using Reexport
using DocStringExtensions
using Random

using RecursiveArrayTools
using LinearAlgebra
using SciMLBase
using SciMLStructures
using SymbolicIndexingInterface
import SciMLStructures as SS 

# TODO We need to set this using Preferences 
const MAXBINSIZE = 100

using OhMyThreads
using Distributed

using LuxCore
using Functors

# For evaluation
mythreadmap(::EnsembleSerial, args...) = map(args...)
mythreadmap(::EnsembleThreads, args...) = tmap(args...)
mythreadmap(::EnsembleDistributed, args...) = pmap(args...)

# General methods for Corleone Layer
get_block_structure(layer::LuxCore.AbstractLuxLayer; kwargs...) = [0, LuxCore.parameterlength(layer)]
get_bounds(layer::LuxCore.AbstractLuxLayer; kwargs...) = (
    get_lower_bound(layer), get_upper_bound(layer),
)
to_val(::T, val) where {T <: Number} = T(val)
to_val(x::AbstractArray{T}, val) where {T <: Number} = T(val) .+ zero(x)
to_val(x::Tuple, val) = tuple(to_val.(x, val)...) 
get_lower_bound(layer::AbstractLuxLayer) = Functors.fmapstructure(Base.Fix2(to_val, -Inf), LuxCore.initialparameters(Random.default_rng(), layer))
get_upper_bound(layer::AbstractLuxLayer) = Functors.fmapstructure(Base.Fix2(to_val, Inf), LuxCore.initialparameters(Random.default_rng(), layer))

# Remakebuffer wrap
# Resolves a single symbolic or integer index to an integer position in its buffer.
function _resolve_index(sys, idx)
    if (i = variable_index(sys, idx)) !== nothing
        return i
    elseif (i = parameter_index(sys, idx)) !== nothing
        return i
    else
        return idx  # already an integer index
    end
end

"""
    __remake_wrap(sys, oldbuffer, idxs, vals)

Non-mutating drop-in for `SymbolicIndexingInterface.remake_buffer` that is
differentiable with Zygote and ReverseDiff.  Instead of calling `setindex!`,
the result is built as

    result = oldbuffer .* keep_flags + Σᵢ eᵢ * valᵢ

where `keep_flags` is a fixed Boolean mask (not differentiated) and `eᵢ` is
the one-hot basis vector for position `i`.  Both `oldbuffer` and `vals` are
full participants in the AD graph.
"""
function __remake_wrap(sys, oldbuffer::AbstractVector, idxs, vals)
    isempty(idxs) && return oldbuffer

    # Resolve symbolic indices → integer positions (non-differentiable)
    int_idxs = map(idx -> _resolve_index(sys, idx), idxs)
    n = length(oldbuffer)

    # Zero out positions that will be replaced (non-differentiable mask)
    keep_flags = [i ∉ int_idxs for i in 1:n]
    result = oldbuffer .* keep_flags

    # Scatter-add each new value via one-hot basis vector
    for (pos, val) in zip(int_idxs, vals)
        eᵢ = [i == pos ? one(eltype(result)) : zero(eltype(result)) for i in 1:n]
        result = result .+ eᵢ .* val
    end

    return result
end

# Fallback for non-AbstractVector buffers (tuples, Nothing, …): delegate to the
# mutating upstream version since these don't flow through the AD graph anyway.
__remake_wrap(sys, p, idxs, vals) = isempty(idxs) ? p : remake_buffer(sys, p, idxs, vals)

is_shooting_layer(x) = false

# Random
_random_value(rng::Random.AbstractRNG, lb::AbstractVector, ub::AbstractVector) = lb .+ rand(rng, eltype(lb), size(lb)...) .* (ub .- lb)

export get_bounds

include("trajectory.jl")
export Trajectory

include("parameters.jl")
export ProblemRemaker
export PiecewiseConstantControl

include("single_shooting.jl")
export SingleShootingLayer 
export get_block_structure 

include("parallel_shooting.jl")
export ParallelShootingLayer
export MultipleShootingLayer

#include("multiple_shooting.jl")
#export MultipleShootingLayer
#export default_initialization
#include("node_initialization.jl")
#export random_initialization, forward_initialization, linear_initialization
#export custom_initialization, constant_initialization, hybrid_initialization

#abstract type AbstractCorleoneFunctionWrapper end

#include("dynprob.jl")
#export CorleoneDynamicOptProblem

end
