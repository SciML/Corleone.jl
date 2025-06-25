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

# The main symbolic metadata structure
include("metadata.jl")
export UncertainParameter, is_uncertain
include("utils.jl")
# This stores the piecewise constant struct and the function which extends controls
include("control_formulations/abstract.jl")
include("control_formulations/directcallback.jl")
export DirectControlCallback
include("control_formulations/searchindex.jl")
export SearchIndexControl
include("control_formulations/ifelsecontrol.jl")
export IfElseControl
include("control_formulations/tanhapproximation.jl")
export TanhControl

include("shooting.jl")
export ShootingGrid
include("predictor.jl")
export OCPredictor

include("node_initialization.jl")
export DefaultsInitialization, ForwardSolveInitialization, RandomInitialization
export LinearInterpolationInitialization, CustomInitialization, ConstantInitialization
export HybridInitialization

include("criteria.jl")
export AbstractOEDCriterion
export ACriterion, DCriterion, ECriterion
export FisherACriterion, FisherDCriterion

include("experimental_design.jl")
export OEDProblemBuilder, InformationGain

include("variable_substitution.jl")
export OCProblemBuilder


end
