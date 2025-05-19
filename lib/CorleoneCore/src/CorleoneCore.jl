module CorleoneCore

using Reexport
using DocStringExtensions

using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D

using SciMLBase
using SciMLStructures
using SymbolicIndexingInterface

using Accessors

# The main symbolic metadata structure
include("metadata.jl")

include("symbolic_operations.jl")
export ∫
export ∀

#include("extend_functions.jl")
include("extend_costs.jl")
export extend_costs

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

# Convert a solution into a trajectory
include("trajectory.jl")
export Trajectory

# Define observed functions 
include("observed.jl")

# Here we prepare the objective and constraints
include("prepare_expression.jl")

end
