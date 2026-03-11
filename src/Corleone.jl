module Corleone

using Reexport
using DocStringExtensions
using Random

using RecursiveArrayTools
using LinearAlgebra
using SciMLBase
using SciMLStructures
using SymbolicIndexingInterface

using OhMyThreads
using Distributed

using ChainRulesCore

using LuxCore
using Functors

# For evaluation
"""
$(SIGNATURES)

Evaluate a mapping operation according to the selected SciML ensemble execution mode.
"""
mythreadmap(::EnsembleSerial, args...) = map(args...)
mythreadmap(::EnsembleThreads, args...) = tmap(args...)
mythreadmap(::EnsembleDistributed, args...) = pmap(args...)

# General methods for Corleone Layer
"""
$(SIGNATURES)

Return the cumulative parameter block structure for `layer`.
"""
get_block_structure(layer::LuxCore.AbstractLuxLayer; kwargs...) = [0, LuxCore.parameterlength(layer)]

"""
$(SIGNATURES)

Return lower and upper bounds of `layer` parameters.
"""
get_bounds(layer::LuxCore.AbstractLuxLayer; kwargs...) = (
    get_lower_bound(layer), get_upper_bound(layer),
)

"""
$(SIGNATURES)

Convert `val` to the numeric type `T`.
"""
to_val(::T, val) where {T <: Number} = T(val)

"""
$(SIGNATURES)

Map scalar conversion from `to_val` to all entries in `x`.
"""
to_val(x::AbstractVector, val) = map(Base.Fix2(to_val, val), x)

"""
$(SIGNATURES)

Return an elementwise lower bound vector for `layer`.
"""
get_lower_bound(layer::AbstractLuxLayer) = Functors.fmapstructure(Base.Fix2(to_val, -Inf), LuxCore.initialparameters(Random.default_rng(), layer))

"""
$(SIGNATURES)

Return an elementwise upper bound vector for `layer`.
"""
get_upper_bound(layer::AbstractLuxLayer) = Functors.fmapstructure(Base.Fix2(to_val, Inf), LuxCore.initialparameters(Random.default_rng(), layer))

"""
$(SIGNATURES)

Return whether `layer` participates in shooting continuity constraints.
"""
is_shooted(layer::AbstractLuxLayer) = false

# Random
"""
$(SIGNATURES)

Sample random values uniformly between elementwise bounds `lb` and `ub`.
"""
_random_value(rng::Random.AbstractRNG, lb::AbstractVector, ub::AbstractVector) = lb .+ rand(rng, eltype(lb), size(lb)...) .* (ub .- lb)

# TODO We need to set this using Preferences
const MAXBINSIZE = 100

include("trajectory.jl")
export Trajectory

include("controls.jl")
export ControlParameter, ControlParameters

include("initializers.jl")
export InitialCondition

include("single_shooting.jl")
export SingleShootingLayer

include("parallel_shooting.jl") 
export ParallelShootingLayer

#include("multiple_shooting.jl")
#export MultipleShootingLayer
#export default_initialization
#include("node_initialization.jl")
#export random_initialization, forward_initialization, linear_initialization
#export custom_initialization, constant_initialization, hybrid_initialization
#
#abstract type AbstractCorleoneFunctionWrapper end
#
#include("dynprob.jl")
#export CorleoneDynamicOptProblem

end
