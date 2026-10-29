module CorleoneOED

using ChainRulesCore: @non_differentiable
using Reexport
@reexport using Corleone
@reexport using Symbolics
using ADTypes: AutoForwardDiff
using LuxCore
using Random
using SymbolicIndexingInterface
using SciMLStructures
using ForwardDiff
using SciMLBase
using DocStringExtensions
using LinearAlgebra

include("measurements.jl")
export AbstractMeasurement, DiscreteMeasurement, ContinuousMeasurement
include("augmentation.jl")
include("oed.jl")
export OEDLayer
export fisher_information, observed_equations, sensitivities
export local_information_gain, global_information_gain
export sampling_sums, sampling_sums!, discrete_sampling_sums, continuous_sampling_sums
export update_fim
include("multiexperiments.jl")
export MultiExperimentLayer
include("criteria.jl")
include("optimization.jl")
export AbstractCriterion, ACriterion, DCriterion, ECriterion
export FisherACriterion, FisherECriterion, FisherDCriterion

end
