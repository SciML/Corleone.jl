module CorleoneCore

using Reexport
using DocStringExtensions
using Random

using SciMLBase
using Accessors
using LinearAlgebra

using LuxCore
import FindFirstFunctions: searchsortedfirstcorrelated, searchsortedlastcorrelated

# Defines a consistent getindex for the 

"""
$(FUNCTIONNAME)

Indicator if a [`AbstractTimeGridLayer`](@ref) omits `tstops``. Returns a `Bool`.
"""
has_tstops(::Any) = false

is_extension_loaded(::Val) = false

# Defines approximators for signals
# TODO Add Linear, Quadratic and stuff here
include("approximators/piecewiseconstant.jl")
export PiecewiseConstant
include("approximators/container.jl")
export SignalContainer 

include("wrapper.jl")
export Wrapper

end
