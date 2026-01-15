"""
$(TYPEDEF)
Generalization of OEDLayer to multiple experiments that can be jointly optimized.
# Fields
$(FIELDS)
"""
struct MultiExperimentLayer{DISCRETE,FIXED,SPLIT,L,P} <: LuxCore.AbstractLuxLayer
    "Layers defining multiexperiments"
    layers::L
    "Number of experiments"
    n_exp::Int
    "Parameters considered in the different experiments"
    params::P
end

function Base.show(io::IO, oed::MultiExperimentLayer{DISCRETE,FIXED,SPLIT}) where {DISCRETE,FIXED,SPLIT}
    (; n_exp) = oed
    type_color, no_color = SciMLBase.get_colorizers(io)
    measurement_text = DISCRETE ? "discrete " : "continuous "

    print(io,
        type_color, "MultiExperimentLayer ", no_color, "with ",
        type_color, measurement_text,
        no_color, "measurement model and ",
        no_color, n_exp,
        no_color, " experiments."
    )
end

function MultiExperimentLayer{DISCRETE}(prob::DEProblem, alg::DEAlgorithm, nexp::Int;
                params=eachindex(prob.p), measurements=[], observed=default_observed, kwargs...) where {DISCRETE}
    layer = OEDLayer{DISCRETE}(prob, alg; params=params, measurements=measurements, observed=observed, kwargs...)
    fixed = is_fixed(layer)
    MultiExperimentLayer{DISCRETE, fixed, false, typeof(layer), typeof(params)}(layer, nexp, params)
end

function MultiExperimentLayer{DISCRETE}(prob::DEProblem, alg::DEAlgorithm, params::AbstractVector{<:AbstractVector{<:Int}}; measurements=[], observed=default_observed, kwargs...) where {DISCRETE}
    nexp = length(params)
    layers = map(params) do param
        OEDLayer{DISCRETE}(prob, alg; params=param, measurements=measurements, observed=observed, kwargs...)
    end
    fixed = all(is_fixed.(layers))
    MultiExperimentLayer{DISCRETE, fixed, true, typeof(layers), typeof(params)}(layers, nexp, params)
end

function MultiExperimentLayer{DISCRETE}(prob::DEProblem, alg::DEAlgorithm, shooting_points::AbstractVector{<:Real}, nexp::Int; params=eachindex(prob.p), measurements=[], observed=default_observed, kwargs...) where {DISCRETE}
    layers = OEDLayer{DISCRETE}(prob, alg, shooting_points...; params=param, measurements=measurements, observed=observed, kwargs...)
    fixed = false
    MultiExperimentLayer{DISCRETE, false, false, typeof(layers), typeof(params)}(layers, nexp, params)
end

function MultiExperimentLayer{DISCRETE}(prob::DEProblem, alg::DEAlgorithm, shooting_points::AbstractVector{<:Real}, params::AbstractVector{<:AbstractVector{<:Int}}=[eachindex(prob.p) for _ in 1:nexp]; measurements=[], observed=default_observed, kwargs...) where {DISCRETE}
    nexp = length(params)
    layers = map(params) do param
        OEDLayer{DISCRETE}(prob, alg, shooting_points...; params=param, measurements=measurements, observed=observed, kwargs...)
    end
    fixed = false
    MultiExperimentLayer{DISCRETE, false, true, typeof(layers), typeof(params)}(layers, nexp, params)
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, multi::MultiExperimentLayer{<:Any, <:Any, true})
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multi.n_exp])
    exp_ps = Tuple(map(1:multi.n_exp) do i
        LuxCore.initialparameters(rng, multi.layers[i])
    end)
    return NamedTuple{exp_names}(exp_ps)
end


function LuxCore.initialparameters(rng::Random.AbstractRNG, multi::MultiExperimentLayer{<:Any, <:Any, false})
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multi.n_exp])
    exp_ps = Tuple([LuxCore.initialparameters(rng, multi.layers) for _ in 1:multi.n_exp])
    return NamedTuple{exp_names}(exp_ps)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, multi::MultiExperimentLayer{<:Any, <:Any, true})
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multi.n_exp])
    exp_ps = Tuple(map(1:multi.n_exp) do i
        LuxCore.initialstates(rng, multi.layers[i])
    end)
    return NamedTuple{exp_names}(exp_ps)
end


function LuxCore.initialstates(rng::Random.AbstractRNG, multi::MultiExperimentLayer{<:Any, <:Any, false})
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multi.n_exp])
    exp_ps = Tuple([LuxCore.initialstates(rng, multi.layers) for _ in 1:multi.n_exp])
    return NamedTuple{exp_names}(exp_ps)
end


function (layer::MultiExperimentLayer{<:Any, <:Any, false})(x, ps, st)
    sols = map(1:layer.n_exp) do i
        ps_local, st_local = getproperty(ps, Symbol("experiment_$i")), getproperty(st, Symbol("experiment_$i"))
        sol, _ = layer.layers(x, ps_local, st_local)
        sol
    end
    return sols, st
end

function (layer::MultiExperimentLayer{<:Any, <:Any, true})(x, ps, st)
    sols = map(enumerate(layer.layers)) do (i,_layer)
        ps_local, st_local = getproperty(ps, Symbol("experiment_$i")), getproperty(st, Symbol("experiment_$i"))
        sol, _ = _layer(x, ps_local, st_local)
        sol
    end
    return sols, st
end

function get_sampling_sums(multi::MultiExperimentLayer{<:Any, <:Any, true}, x, ps, st::NamedTuple{fields}) where {fields}
    reduce(vcat, map(zip(multi.layers,fields)) do (layer,field)
        get_sampling_sums(layer, x, getproperty(ps, field), getproperty(st, field))
    end)
end

function get_sampling_sums(multi::MultiExperimentLayer{<:Any, <:Any, false}, x, ps,st::NamedTuple{fields}) where {fields}
    reduce(vcat, map(fields) do field
        get_sampling_sums(multi.layers, x, getproperty(ps, field), getproperty(st, field))
    end)
end


function fisher_information(multi::MultiExperimentLayer{<:Any, <:Any, false}, x, ps, st::NamedTuple{fields}) where {fields}
    sum(map(fields) do field
        fisher_information(multi.layers, x, getproperty(ps, field), getproperty(st, field))[1]
    end), st
end

function fisher_information(multi::MultiExperimentLayer{<:Any, <:Any, true}, x, ps, st::NamedTuple{fields}) where {fields}
    map(enumerate(fields)) do (i,field)
        fisher_information(multi.layers[i], x, getproperty(ps, field), getproperty(st, field))[1]
    end, st
end



get_bounds(layer::MultiExperimentLayer{<:Any, <:Any, true}) = begin
    exp_names = Tuple([Symbol("experiment_$i") for i=1:layer.n_exp])
    exp_bounds = map(Tuple(1:layer.n_exp)) do i
        get_bounds(layer.layers[i])
    end
    NamedTuple{exp_names}(first.(exp_bounds)), NamedTuple{exp_names}(last.(exp_bounds))
end

get_bounds(layer::MultiExperimentLayer{<:Any, <:Any, false}) = begin
    exp_names = Tuple([Symbol("experiment_$i") for i=1:layer.n_exp])
    exp_bounds = map(Tuple(1:layer.n_exp)) do i
        get_bounds(layer.layers)
    end
    NamedTuple{exp_names}(first.(exp_bounds)), NamedTuple{exp_names}(last.(exp_bounds))
end

"""
$(METHODLIST)
Computes the block structure as defined by the `MultiExperimentLayer`, which may come from
two levels: 1) the different experiments, and 2) multiple shooting discretizations on the
experiment level.
"""
function get_block_structure(layer::MultiExperimentLayer{<:Any, true})
    blocks = map(layer.layers) do _layer
        get_block_structure(_layer)
    end |> Tuple

    for i=1:layer.n_exp-1
        blocks[i+1] .= blocks[i+1] .+ blocks[i][end]
    end
    block_structure = reduce(vcat, [i == 1 ? blocks[i] : blocks[i][2:end] for i =1:layer.n_exp])

    return block_structure
end

function get_block_structure(layer::MultiExperimentLayer{<:Any, false})
    blocks = map(1:layer.n_exp) do i
        get_block_structure(layer.layers)
    end |> Tuple

    for i=1:layer.n_exp-1
        blocks[i+1] .= blocks[i+1] .+ blocks[i][end]
    end
    block_structure = reduce(vcat, [i == 1 ? blocks[i] : blocks[i][2:end] for i =1:layer.n_exp])

    return block_structure
end