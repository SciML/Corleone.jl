module Corleone

using Reexport
using DocStringExtensions
import PrecompileTools: @compile_workload, @setup_workload
using Random

using RecursiveArrayTools
using LinearAlgebra
using SciMLBase
using SciMLStructures
using SymbolicIndexingInterface

using OhMyThreads
using Distributed

using ChainRulesCore

using LuxCore
using Functors

# For evaluation
mythreadmap(::EnsembleSerial, args...) = map(args...)
mythreadmap(::EnsembleThreads, args...) = tmap(args...)
mythreadmap(::EnsembleDistributed, args...) = pmap(args...)

"""
    get_block_structure(layer; kwargs...)

Return cumulative parameter boundaries for a Corleone layer. The first entry is
zero and the final entry is the number of optimization parameters. Layer wrappers
should extend this method when their parameter blocks have additional structure.
"""
get_block_structure(layer::LuxCore.AbstractLuxLayer; kwargs...) = [0, LuxCore.parameterlength(layer)]

"""
    get_bounds(layer; kwargs...)

Return lower and upper bounds for the optimization variables of `layer` as a tuple
`(lower, upper)`. A layer extension must preserve the same structure in both values.
"""
get_bounds(layer::LuxCore.AbstractLuxLayer; kwargs...) = (
    get_lower_bound(layer), get_upper_bound(layer),
)
to_val(::T, val) where {T <: Number} = T(val)
to_val(x::AbstractArray{T}, val) where {T <: Number} = T(val) .+ zero(x)
get_lower_bound(layer::AbstractLuxLayer) = Functors.fmapstructure(Base.Fix2(to_val, -Inf), LuxCore.initialparameters(Random.default_rng(), layer))
get_upper_bound(layer::AbstractLuxLayer) = Functors.fmapstructure(Base.Fix2(to_val, Inf), LuxCore.initialparameters(Random.default_rng(), layer))

# Random
_random_value(rng::Random.AbstractRNG, lb::AbstractVector, ub::AbstractVector) = lb .+ rand(rng, eltype(lb), size(lb)...) .* (ub .- lb)

include("trajectory.jl")
export Trajectory

include("local_controls.jl")
export ControlParameter

include("single_shooting.jl")
export SingleShootingLayer
include("multiple_shooting.jl")
export MultipleShootingLayer
export default_initialization
include("node_initialization.jl")
export random_initialization, forward_initialization, linear_initialization
export custom_initialization, constant_initialization, hybrid_initialization

abstract type AbstractCorleoneFunctionWrapper end

include("dynprob.jl")
export CorleoneDynamicOptProblem

include("multistage.jl")
export StageTransition

@setup_workload begin
    @compile_workload begin
        rng = Random.MersenneTwister(1)
        controls = ControlParameter(
            collect(0.0:0.25:0.75);
            name = :u,
            controls = [0.1, 0.2, 0.3, 0.4],
            bounds = (0.0, 1.0),
        )
        get_timegrid(controls, (0.0, 0.75))
        control_length(controls)
        get_controls(rng, controls)
        get_bounds(controls)
        check_consistency(rng, controls)
        build_index_grid(controls; tspan = (0.0, 0.75))
    end
end

end
