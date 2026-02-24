struct ParallelShootingLayer{L <: NamedTuple, A <: SciMLBase.EnsembleAlgorithm} <: LuxCore.AbstractLuxWrapperLayer{:layers}
    name::Symbol
    "The layers to be solved in parallel. Each layer should be a SingleShootingLayer."
    layers::L
    "The underlying ensemble algorithm to use for parallelization. Default is `EnsembleThreads`."
    ensemble_algorithm::A
end

ParallelShootingLayer(layers::NamedTuple; kwargs...) = ParallelShootingLayer(
    get(kwargs, :name, gensym(:parallel_shooting)), 
    layers, 
    get(kwargs, :ensemble_algorithm, EnsembleSerial()))

function ParallelShootingLayer(layers::AbstractLuxLayer...; kwargs...)
    @assert all(is_shooting_layer, layers) "All layers must be shooting layers."
    layers = NamedTuple{ntuple(i->Symbol(:layer,i), length(layers))}(layers)
    ParallelShootingLayer(layers; kwargs...)
end

function get_block_structure(layer::ParallelShootingLayer)
    return vcat(0, cumsum(map(LuxCore.parameterlength, layer.layers)))
end

function (layer::ParallelShootingLayer)(u0, ps, st)
    _parallel_solve(layer.ensemble_algorithm, layer.layers, u0, ps, st)
end

__getidx(x, id) = x[id]
__getidx(x::NamedTuple, id) = getproperty(x, id)

function _parallel_solve(
        alg::SciMLBase.EnsembleAlgorithm,
        layers::NamedTuple{fields},
        u0,
        ps,
        st::NamedTuple{fields},
    ) where {fields}

    args = ntuple(
            i -> (__getidx(layers, fields[i]), u0, __getidx(ps, fields[i]), __getidx(st, fields[i])), length(st)
        )
    
    ret =  mythreadmap(alg, Base.splat(LuxCore.apply), args)
    return NamedTuple{fields}(first.(ret)), NamedTuple{fields}(last.(ret))
end

function SciMLBase.remake(layer::ParallelShootingLayer; kwargs...)
    layers = map(keys(layer.layers)) do k 
        layer_kwargs = get(kwargs, k, kwargs)
        k, remake(layer.layers[k]; layer_kwargs...)
    end |> NamedTuple
    ensemble_algorithm = get(kwargs, :ensemble_algorithm, layer.ensemble_algorithm)
    ParallelShootingLayer(layer.name, layers, ensemble_algorithm)
end

"""
$(TYPEDEF)

Defines a layer for multiple shooting. Simply a wrapper for the [ParallelShootingLayer](@ref) but returns a single trajectory.
"""
struct MultipleShootingLayer{L} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    "The instance of a [ParallelShootingLayer](@ref) to be solved in parallel."
    layer::L 
end

function MultipleShootingLayer(layer::LuxCore.AbstractLuxLayer, shooting_points::Real...; kwargs...)
    @assert is_shooting_layer(layer) "The provided layer must be a shooting layer."
    problem = get_problem(layer)
    tspan = get_tspan(layer)
    quadratures = get_quadrature_indices(layer)
    tunables = setdiff(variable_symbols(problem), quadratures)
    tpoints = unique!(sort!(vcat(collect(shooting_points), collect(tspan))))
    layers = ntuple(i -> remake(layer, 
        tspan = (tpoints[i], tpoints[i+1]),
        tunable_u0 = i == 1 ? get_tunable_u0(layer) : tunables,
    ), length(tpoints)-1)
    layers = NamedTuple{ntuple(i->Symbol(:layer,i), length(layers))}(layers) 
    layer = ParallelShootingLayer(layers; kwargs...)
    MultipleShootingLayer{typeof(layer)}(layer)
end

function SciMLBase.remake(layer::MultipleShootingLayer; kwargs...)
    layer = remake(layer.layer; kwargs...)
    MultipleShootingLayer{typeof(layer)}(layer)
end

function (layer::MultipleShootingLayer)(u0, ps, st)
    results, st = layer.layer(u0, ps, st)
    return results, st
end
