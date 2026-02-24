struct ParallelShootingLayer{L <: NamedTuple, A <: SciMLBase.EnsembleAlgorithm} <: LuxCore.AbstractLuxWrapperLayer{:layers}
    name::Symbol
    "The layers to be solved in parallel. Each layer should be a SingleShootingLayer."
    layers::L
    "The underlying ensemble algorithm to use for parallelization. Default is `EnsembleThreads`."
    ensemble_algorithm::A
end

function ParallelShootingLayer(layer::SingleShootingLayer, shooting_points::Real...; name = gensym(:parallel_shooting), ensemble_algorithm::SciMLBase.EnsembleAlgorithm = EnsembleThreads())
    shooting_points = unique!(sort!(vcat(collect(shooting_points), collect(get_tspan(layer)))))
     # We need to add the initial time and the final time to the shooting points
    tspans = collect((t0, tinf) for (t0, tinf) in zip(shooting_points[1:end-1], shooting_points[2:end]))
    layers = map(eachindex(tspans)) do i
        clamp_tspan(layer, tspans[i])
    end 
    layers = NamedTuple{ntuple(i->Symbol(:layer,i), length(layers))}(layers)
    ParallelShootingLayer(name, layers, ensemble_algorithm)
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