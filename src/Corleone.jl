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

include("solutions/Solutions.jl")

include("layers/Layers.jl")

# Bridge: add Trajectory dispatch to Corleone.shooting_constraints (which was shadowed
# by the Layers definition) so both the layer and trajectory APIs share one name.
shooting_constraints(traj::Solutions.Trajectory) = Solutions.shooting_constraints(traj)
shooting_constraints!(res::AbstractVector, traj::Solutions.Trajectory) = Solutions.shooting_constraints!(res, traj)

end
