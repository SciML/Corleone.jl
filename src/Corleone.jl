module Corleone

using Reexport
using DocStringExtensions
using Random

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

using SciMLBase
using SciMLStructures
using SymbolicIndexingInterface
using Accessors
using LinearAlgebra

"""
$(TYPEDEF)

Abstract type defining different formulation for piecewise constant control signals.

All discrete implementations take in a list of variables and time points, which are in turn extended.
"""
abstract type AbstractControlFormulation end

"""
$(TYPEDEF)

An abstract builder which executes a series of transformations to transform the dynamical system and additional specifications into an optimal control problem.
"""
abstract type AbstractBuilder end

"""
$(TYPEDEF)

Abstract type defining different formulations for initialization of shooting node variables.

"""
abstract type AbstractNodeInitialization end

function (f::AbstractNodeInitialization)(problem::SciMLBase.AbstractSciMLProblem, args...; kwargs...)
    throw(ArgumentError("The initialization $f is not implemented."))
end

# The main symbolic metadata structure
include("metadata.jl")
export UncertainParameter, is_uncertain
include("utils.jl")


# This stores the piecewise constant struct and the function which extends controls
#include("control_formulations/abstract.jl")
#include("control_formulations/directcallback.jl")
#export DirectControlCallback
#include("control_formulations/searchindex.jl")
#export SearchIndexControl
#include("control_formulations/ifelsecontrol.jl")
#export IfElseControl
#include("control_formulations/tanhapproximation.jl")
#export TanhControl

#include("shooting.jl")
#export ShootingGrid
#include("predictor.jl")
#export OCPredictor

#include("initialization/node_initialization.jl")
#export DefaultsInitialization, ForwardSolveInitialization, RandomInitialization
#export LinearInterpolationInitialization, CustomInitialization, ConstantInitialization
#export HybridInitialization

#include("criteria.jl")
#export AbstractOEDCriterion
#export ACriterion, DCriterion, ECriterion
#export FisherACriterion, FisherDCriterion

#include("builders/abstract.jl")
#include("builders/variable_substitution.jl")
#include("builders/optimalcontrolfunction.jl")
#include("builders/ocbuilder.jl")
#export OCProblemBuilder
#include("builders/oedbuilder.jl")
#export OEDProblemBuilder

#include("analysis/information_criteria.jl")
#export InformationGain

end
