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

include("oed.jl")
export OEDLayer
export fisher_information, observed_equations, sensitivities
export local_information_gain, global_information_gain

include("criteria.jl")
export ACriterion, DCriterion, ECriterion
export FisherACriterion, FisherECriterion, FisherDCriterion

end
