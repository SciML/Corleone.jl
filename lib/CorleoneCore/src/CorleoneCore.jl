module CorleoneCore

using DocStringExtensions
using LuxCore
using SciMLBase
using Functors
using Setfield
using Random
using Reexport

include("./InternalWrapper.jl")

# Dispatch on an Abstract Layer of Lux 
# TSTOPS, SAVEATS <: Bool indicates the use of these timepoints

"""
$(TYPEDEF)

A subtype for an `AbstractLuxLayer` which indicates the use of `tstops` or `saveat`.
"""
abstract type AbstractTimeGridLayer{TSTOPS,SAVEATS} <: LuxCore.AbstractLuxLayer end

"""
$(FUNCTIONNAME)

Indicator if a [`AbstractTimeGridLayer`](@ref) omits `tstops``. Returns a `Bool`.
"""
has_tstops(::AbstractTimeGridLayer{TSTOPS}) where {TSTOPS} = TSTOPS
has_tstops(::Any) = false

"""
$(FUNCTIONNAME)

Indicator if a [`AbstractTimeGridLayer`](@ref) omits `saveat`s. Returns a `Bool`.
"""
has_saveats(::AbstractTimeGridLayer{<:Any,SAVEATS}) where {SAVEATS} = SAVEATS
has_saveats(::Any) = false

# Initialization 
InternalWrapper.initialize_model(f::SciMLBase.AbstractDiffEqFunction, args...) = f


# Common utility functions 
include("utils.jl")
export collect_saveat, collect_tstops
export contains_timegrid_layer
export contains_tstop_layer
export contains_saveat_layer

# The basic for useage of gridded parameters
include("grid_parameters.jl")
export Parameter
include("parameters.jl")
export ParameterContainer
include("grid_function.jl")
export GridFunction
include("simulation_grid.jl")
export SimulationGrid
include("model.jl")
export DynamicModel

# Similar to LuxCore.Internal we define extensions for wrapping models here
end
