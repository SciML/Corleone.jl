module Corleone

using Reexport
using DocStringExtensions
using DispatchDoctor: @stable
using Random
using LinearAlgebra

using ConcreteStructs

using SciMLBase
using SciMLStructures
using SymbolicIndexingInterface



using ChainRulesCore

using LuxCore
using Functors

#include("core/Core.jl")

include("solutions/Solutions.jl")

include("layers/Layers.jl")

# Bridge: add Trajectory dispatch to Corleone.shooting_constraints (which was shadowed
# by the Layers definition) so both the layer and trajectory APIs share one name.
shooting_constraints(traj::Solutions.Trajectory) = Solutions.shooting_constraints(traj)
shooting_constraints!(res::AbstractVector, traj::Solutions.Trajectory) = Solutions.shooting_constraints!(res, traj)

#=

include("trajectory.jl")
export Trajectory

include("local_controls.jl")
export ControlParameter

include("single_shooting.jl")
export SingleShootingLayer
include("multiple_shooting.jl")
export MultipleShootingLayer
export default_initialization
include("node_initialization.jl")
export random_initialization, forward_initialization, linear_initialization
export custom_initialization, constant_initialization, hybrid_initialization

abstract type AbstractCorleoneFunctionWrapper end

include("dynprob.jl")
export CorleoneDynamicOptProblem
=#
end
