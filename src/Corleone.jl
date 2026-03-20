module Corleone

using Reexport
using DocStringExtensions
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

using RuntimeGeneratedFunctions
RuntimeGeneratedFunctions.init(@__MODULE__)

# For evaluation
"""
$(SIGNATURES)

Evaluate a mapping operation according to the selected SciML ensemble execution mode.
"""
mythreadmap(::EnsembleSerial, args...) = map(args...)
mythreadmap(::EnsembleThreads, args...) = tmap(args...)
mythreadmap(::EnsembleDistributed, args...) = pmap(args...)

# General methods for Corleone Layer
"""
$(SIGNATURES)

Return the cumulative parameter block structure for `layer`.
"""
get_block_structure(layer::LuxCore.AbstractLuxLayer; kwargs...) = [0, LuxCore.parameterlength(layer)]

get_timegrid(::Any) = []

get_timegrid(layer::LuxCore.AbstractLuxWrapperLayer{LAYER}) where {LAYER} = get_timegrid(getfield(layer, LAYER))
get_timegrid(layer::LuxCore.AbstractLuxContainerLayer{LAYERS}) where {LAYERS} = begin
    vals = map(LAYERS) do LAYER
        get_timegrid(getfield(layer, LAYER))
    end
    NamedTuple{LAYERS}(vals)
end
get_timegrid(nt::NamedTuple) = map(get_timegrid, nt)


"""
$(SIGNATURES)

Return lower and upper bounds of `layer` parameters.
"""
get_bounds(x) = (get_lower_bound(x), get_upper_bound(x))

"""
$(SIGNATURES)

Convert `val` to the numeric type `T`.
"""
_to_val(::T, val) where {T <: Number} = T(val)
_to_val(x, val) = map(x) do xi
    _to_val(xi, val)
end
"""
$(SIGNATURES)

Map scalar conversion from `to_val` to all entries in `x`.
"""
to_val(x, val) = fmap(Base.Fix2(_to_val, val), x)

"""
$(SIGNATURES) 

Compute the number of shooting constraints of a `AbstractLuxLayer`.
"""
get_number_of_shooting_constraints(layer::LuxCore.AbstractLuxLayer) = 0
get_number_of_shooting_constraints(nt::NamedTuple) = sum(get_number_of_shooting_constraints, values(nt))
get_number_of_shooting_constraints(layer::LuxCore.AbstractLuxWrapperLayer{LAYER}) where {LAYER} = get_number_of_shooting_constraints(getfield(layer, LAYER))
get_number_of_shooting_constraints(layer::LuxCore.AbstractLuxContainerLayer{LAYERS}) where {LAYERS} = sum(
    map(LAYERS) do LAYER
        get_number_of_shooting_constraints(getfield(layer, LAYER))
    end
)

"""
$(SIGNATURES)

Return an elementwise lower bound vector for `layer`.
"""
get_lower_bound(layer::AbstractLuxLayer) = Functors.fmapstructure(Base.Fix2(to_val, -Inf), LuxCore.initialparameters(Random.default_rng(), layer))
get_lower_bound(layer::LuxCore.AbstractLuxWrapperLayer{LAYER}) where {LAYER} = get_lower_bound(getfield(layer, LAYER))
get_lower_bound(layer::LuxCore.AbstractLuxContainerLayer{LAYERS}) where {LAYERS} = NamedTuple{LAYERS}(
    map(LAYERS) do LAYER
        getfield(layer, LAYER) |> get_lower_bound
    end
)
get_lower_bound(nt::Union{NamedTuple, Tuple}) = map(get_lower_bound, nt)

"""
$(SIGNATURES)

Return an elementwise upper bound vector for `layer`.
"""
get_upper_bound(layer::AbstractLuxLayer) = Functors.fmapstructure(Base.Fix2(to_val, Inf), LuxCore.initialparameters(Random.default_rng(), layer))
get_upper_bound(layer::LuxCore.AbstractLuxWrapperLayer{LAYER}) where {LAYER} = get_upper_bound(getfield(layer, LAYER))
get_upper_bound(layer::LuxCore.AbstractLuxContainerLayer{LAYERS}) where {LAYERS} = NamedTuple{LAYERS}(
    map(LAYERS) do LAYER
        getfield(layer, LAYER) |> get_upper_bound
    end
)
get_upper_bound(nt::Union{NamedTuple, Tuple}) = map(get_upper_bound, nt)

"""
$(SIGNATURES)

Return whether `layer` participates in shooting continuity constraints.
"""
is_shooted(layer::AbstractLuxLayer) = false

# Random
"""
$(SIGNATURES)

Sample random values uniformly between elementwise bounds `lb` and `ub`.
"""
_random_value(rng::Random.AbstractRNG, lb::AbstractVector, ub::AbstractVector) = lb .+ rand(rng, eltype(lb), size(lb)...) .* (ub .- lb)

# TODO We need to set this using Preferences
const MAXBINSIZE = 100

include("trajectory.jl")
export Trajectory

include("controls.jl")
export ControlParameter, FixedControlParameter
export ControlParameters

include("initializers.jl")
export InitialCondition

include("single_shooting.jl")
export SingleShootingLayer

include("parallel_shooting.jl")
export ParallelShootingLayer

include("multiple_shooting.jl")
export MultipleShootingLayer

include("dynprob.jl")
export DynamicOptimizationLayer


#export default_initialization
#include("node_initialization.jl")
#export random_initialization, forward_initialization, linear_initialization
#export custom_initialization, constant_initialization, hybrid_initialization

end
