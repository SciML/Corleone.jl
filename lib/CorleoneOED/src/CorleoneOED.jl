module CorleoneOED

using Reexport
@reexport using Corleone
@reexport using Symbolics
using LuxCore
using Random
using SymbolicIndexingInterface
using SciMLStructures
using ForwardDiff
using SciMLBase
using DocStringExtensions
using LinearAlgebra

# Core OED implementation
include("augmentation.jl")
include("oed.jl")

# Additional modules
include("multiexperiments.jl")
include("criteria.jl")
export MultiExperimentLayer
export ACriterion, DCriterion, ECriterion
export FisherACriterion, FisherECriterion, FisherDCriterion

# Main exports
export OEDLayer
export fisher_information, observed_equations, sensitivities
export local_information_gain, global_information_gain

end
