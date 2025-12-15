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

# TODO Clean and more tests (also for LV)
include("augmentation.jl")

# TODO Docs
include("oed.jl")
export OEDLayer
export fisher_information, observed_equations, sensitivities
export local_information_gain, global_information_gain

# TODO
# Dispatch for optimality(crit, oed, x, ps, st)
# --> Compute optimality condition for specific crit
# e.g. tr(\Pi) for the ACriterion
include("criteria.jl")
export ACriterion, DCriterion, ECriterion
export FisherACriterion, FisherECriterion, FisherDCriterion

end
