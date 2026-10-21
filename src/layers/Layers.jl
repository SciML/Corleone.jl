using LuxCore
using Random
using ConcreteStructs
using ChainRulesCore

using SciMLBase
using SciMLStructures
using SymbolicIndexingInterface

using DocStringExtensions

is_extension_loaded(::Val) = false

include("./lookup.jl")
include("./parameters.jl")

include("./initials.jl")
export InitialConditions

include("./parameter.jl")
export Parameter

include("./shooting.jl")
export ShootingLayer

@non_differentiable get_timestops(layer, ps, st, type)
