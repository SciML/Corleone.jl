module CorleoneModelingToolkitExtension

import Corleone
import ModelingToolkit
import SciMLBase
using Corleone: ShootingLayer, Tunable, Constant
using Corleone.DocStringExtensions: SIGNATURES
using Corleone.Random: default_rng
using Corleone.SymbolicIndexingInterface: variable_symbols, parameter_symbols, variable_index, parameter_index, default_values, symbolic_type, ArraySymbolic
using ModelingToolkit: Symbolics, SymbolicUtils

include("MTKExtension/utils.jl")
include("MTKExtension/annotations.jl")
include("MTKExtension/optimal_control.jl")

end
