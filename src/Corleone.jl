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
mythreadmap(::EnsembleSerial, args...) = map(args...) 
mythreadmap(::EnsembleThreads, args...) = tmap(args...) 
mythreadmap(::EnsembleDistributed, args...) = pmap(args...) 

# General methods for Corleone Layer 
get_block_structure(layer::LuxCore.AbstractLuxLayer; kwargs...) = [0]
get_bounds(layer::LuxCore.AbstractLuxLayer; kwargs...) = (
	get_lower_bound(layer), get_upper_bound(layer)
)
to_val(::T, val) where T <: Number = T(val) 
to_val(x::AbstractArray{T}, val) where T <: Number = T(val) .+ zero(x)
get_lower_bound(layer::AbstractLuxLayer) = Functors.fmapstructure(Base.Fix2(to_val, -Inf), LuxCore.initialparameters(Random.default_rng(), layer)) 
get_upper_bound(layer::AbstractLuxLayer) = Functors.fmapstructure(Base.Fix2(to_val, Inf), LuxCore.initialparameters(Random.default_rng(), layer)) 

# Random 
_random_value(rng::Random.AbstractRNG, lb::AbstractVector, ub::AbstractVector) = lb .+ rand(rng, size(lb)) .* (ub .- lb)

include("trajectory.jl") 
export Trajectory
export shooting_constraints, shooting_constraints!

include("local_controls.jl")
export ControlParameter

include("single_shooting.jl")
export SingleShootingLayer
include("multiple_shooting.jl")
export MultipleShootingLayer
include("node_initialization.jl")
export default_initialization, random_initialization
export forward_solve
#export DefaultsInitialization, ConstantInitialization
#export LinearInterpolationInitialization, ForwardSolveInitialization
#export HybridInitialization, RandomInitialization, CustomInitialization

#include("abstract.jl")

end
