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
    get(kwargs, :ensemble_algorithm, EnsembleSerial())
)

function ParallelShootingLayer(layers::AbstractLuxLayer...; kwargs...)
    layers = NamedTuple{ntuple(i -> Symbol(:layer, i), length(layers))}(layers)
    return ParallelShootingLayer(layers; kwargs...)
end

function get_block_structure(layer::ParallelShootingLayer)
    return vcat(0, cumsum(map(LuxCore.parameterlength, layer.layers)))
end

function (layer::ParallelShootingLayer)(u0, ps, st)
    return _parallel_solve(layer.ensemble_algorithm, layer.layers, u0, ps, st)
end

@generated function _parallel_solve(
        alg::SciMLBase.EnsembleAlgorithm,
        layers::NamedTuple{fields},
        u0,
        ps,
        st::NamedTuple{fields},
    ) where {fields}
    exprs = Expr[]
    args = [gensym() for f in fields]
    for i in eachindex(fields)
        push!(
            exprs, :(
                $(args[i]) =
                    (layers.$(fields[i]), u0, ps.$(fields[i]), st.$(fields[i]))
            )
        )
    end
    push!(
        exprs, :(
            ret =
                mythreadmap(alg, Base.splat(LuxCore.apply), $(Expr(:tuple, args...)))
        )
    )
    push!(
        exprs, :(
            NamedTuple{$(fields)}(first.(ret)), NamedTuple{$(fields)}(last.(ret)),
        )
    )
    ex = Expr(:block, exprs...)
    return ex
end

function SciMLBase.remake(layer::ParallelShootingLayer; kwargs...)
    layers = map(keys(layer.layers)) do k
        layer_kwargs = get(kwargs, k, kwargs)
        k, remake(layer.layers[k]; layer_kwargs...)
    end |> NamedTuple
    ensemble_algorithm = get(kwargs, :ensemble_algorithm, layer.ensemble_algorithm)
    return ParallelShootingLayer(layer.name, layers, ensemble_algorithm)
end

function get_timestops(layer::ParallelShootingLayer, st::NamedTuple{fields} = LuxCore.initialstates(Random.default_rng(), layer)) where {fields}
    (; layers) = layer
    return map(fields) do f
        f, get_timestops(getproperty(layers, f), getproperty(st, f))
    end |> NamedTuple
end
