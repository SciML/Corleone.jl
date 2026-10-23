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

include("utilities/Utilities.jl")

include("layers/Layers.jl")
export ShootingLayer

@setup_workload begin
    @compile_workload begin
    end
end

end
