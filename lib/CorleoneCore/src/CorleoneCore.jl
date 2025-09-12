module CorleoneCore

using Reexport
using DocStringExtensions
using Random

using RecursiveArrayTools
using LinearAlgebra
using ComponentArrays
using SciMLBase
using SciMLStructures
using SymbolicIndexingInterface

using CommonSolve
using ChainRulesCore

using LuxCore
using Symbolics

include("local_controls.jl")
export ControlParameter

include("single_shooting.jl")
include("multiple_shooting.jl")

include("augmentation.jl")

include("node_initialization.jl")
export DefaultsInitialization, ConstantInitialization
export LinearInterpolationInitialization, ForwardSolveInitialization
export HybridInitialization, RandomInitialization, CustomInitialization


end
