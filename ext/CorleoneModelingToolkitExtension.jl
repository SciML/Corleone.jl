module CorleoneModelingToolkitExtension

@info "Loading MTK Extension"

using Corleone
using ModelingToolkit

using Corleone.LuxCore
using Corleone.Random
using Corleone.DocStringExtensions
using Corleone.SciMLBase

using ModelingToolkit.Symbolics
using ModelingToolkit.SymbolicUtils
using ModelingToolkit.Setfield
using ModelingToolkit.SymbolicIndexingInterface
using ModelingToolkit.Symbolics.RuntimeGeneratedFunctions

RuntimeGeneratedFunctions.init(@__MODULE__)

include("MTKExtension/utils.jl")

include("MTKExtension/optimal_control.jl")

end
