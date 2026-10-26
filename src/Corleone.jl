module Corleone

using Reexport
using DocStringExtensions
import PrecompileTools: @compile_workload, @setup_workload

using LuxCore
using Random
using ConcreteStructs
using ChainRulesCore

using SciMLBase
using SciMLStructures
using SymbolicIndexingInterface
import RecursiveArrayTools

using OhMyThreads
using Distributed

using ChainRulesCore

using LuxCore

# For evaluation
mythreadmap(::EnsembleSerial, args...) = map(args...)
mythreadmap(::EnsembleThreads, args...) = tmap(args...)
mythreadmap(::EnsembleDistributed, args...) = pmap(args...)

# Extensions
is_extension_loaded(::Val) = false

include("optimization.jl")
include("utilities/Utilities.jl")

include("layers/Layers.jl")
export ShootingLayer

export get_bounds, get_lower_bound, get_upper_bound, get_number_of_shooting_constraints
export shooting_constraints, shooting_constraints!

include("deprecation/controls.jl")
include("deprecation/single_shooting.jl")
include("multiple_shooting.jl")
export ControlParameter, SingleShootingLayer, MultipleShootingLayer

@setup_workload begin
    @compile_workload begin
    end
end

end
