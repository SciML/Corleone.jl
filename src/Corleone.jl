module Corleone

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
# using Symbolics

include("trajectory.jl") 
export Trajectory
export shooting_constraints, shooting_constraints!

include("local_controls.jl")
export ControlParameter

include("single_shooting.jl")
export SingleShootingLayer
include("multiple_shooting.jl")
export MultipleShootingLayer

#include("augmentation.jl")

#include("criteria.jl")
#export ACriterion, DCriterion, ECriterion
#export FisherACriterion, FisherDCriterion, FisherECriterion

#include("node_initialization.jl")
#export DefaultsInitialization, ConstantInitialization
#export LinearInterpolationInitialization, ForwardSolveInitialization
#export HybridInitialization, RandomInitialization, CustomInitialization

#include("oed.jl")
#export OEDLayer, get_sampling_constraint
#include("multi_experiments.jl")
#export MultiExperimentLayer
#include("information_gain.jl")
#export InformationGain

#include("abstract.jl")

end
