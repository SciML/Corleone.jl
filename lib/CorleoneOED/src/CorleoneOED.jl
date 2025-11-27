module CorleoneOED

using Reexport 
@reexport using Corleone 
@reexport using Symbolics
using LuxCore
using Random
using SymbolicIndexingInterface
using SciMLStructures

using SciMLBase
using DocStringExtensions
using LinearAlgebra

include("augmentation.jl")
export OEDLayer
include("criteria.jl")
export ACriterion, DCriterion, ECriterion
export FisherACriterion, FisherECriterion, FisherDCriterion
end
