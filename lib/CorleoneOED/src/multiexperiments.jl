"""
$(TYPEDEF)
Generalization of OEDLayer to multiple experiments that can be jointly optimized.
# Fields
$(FIELDS)
"""
struct MultiExperimentLayer{DISCRETE, FIXED, SPLIT, SHOOTING, L, P} <: LuxCore.AbstractLuxLayer
    "Layers defining multiexperiments"
    layers::L
    "Number of experiments"
    n_exp::Int
    "Parameters considered in the different experiments"
    params::P
end

function Base.show(io::IO, oed::MultiExperimentLayer{DISCRETE, FIXED, SPLIT}) where {DISCRETE, FIXED, SPLIT}
    (; n_exp, params) = oed
    type_color, no_color = SciMLBase.get_colorizers(io)
    measurement_text = DISCRETE ? "discrete " : "continuous "
    layer_text = FIXED ? "Fixed " : ""

    print(
        io,
        no_color, layer_text,
        type_color, "MultiExperimentLayer ", no_color, "with ",
        type_color, measurement_text,
        no_color, "measurement model and ",
        no_color, n_exp,
        no_color, " experiments.\n"
    )
    return if SPLIT
        print(
            io,
            no_color, "Considered parameters are split among the experiments:\n"
        )
        [
            print(
                    io, "Experiment $i considers parameters: $param." * (i == length(params.original) ? "" : "\n")
                ) for (i, param) in enumerate(params.original)
        ]
    end
end

function MultiExperimentLayer{DISCRETE}(
        prob::DEProblem, alg::DEAlgorithm, nexp::Int;
        params = eachindex(prob.p), measurements = [], observed = default_observed, kwargs...
    ) where {DISCRETE}
    layer = OEDLayer{DISCRETE}(prob, alg; params = params, measurements = measurements, observed = observed, kwargs...)
    fixed = is_fixed(layer)
    return MultiExperimentLayer{DISCRETE, fixed, false, SingleShootingLayer, typeof(layer), typeof(params)}(layer, nexp, params)
end

function MultiExperimentLayer{DISCRETE}(prob::DEProblem, alg::DEAlgorithm, params::Vector{Vector{Int64}}; measurements = [], observed = default_observed, kwargs...) where {DISCRETE}
    nexp = length(params)
    layers = map(params) do param
        OEDLayer{DISCRETE}(prob, alg; params = param, measurements = measurements, observed = observed, kwargs...)
    end |> Tuple
    fixed = all(is_fixed.(layers))

    all_params = union(params...)
    common = sort(all_params)
    idxmap = Dict(val => i for (i, val) in enumerate(common))

    new_params = (; original = params, all = common, permutation = idxmap)

    return MultiExperimentLayer{DISCRETE, fixed, true, SingleShootingLayer, typeof(layers), typeof(new_params)}(layers, nexp, new_params)
end

function MultiExperimentLayer{DISCRETE}(prob::DEProblem, alg::DEAlgorithm, shooting_points::AbstractVector{<:Real}, nexp::Int; params = eachindex(prob.p), measurements = [], observed = default_observed, kwargs...) where {DISCRETE}
    layers = OEDLayer{DISCRETE}(prob, alg, shooting_points...; params = params, measurements = measurements, observed = observed, kwargs...)
    return MultiExperimentLayer{DISCRETE, false, false, MultipleShootingLayer, typeof(layers), typeof(params)}(layers, nexp, params)
end

function MultiExperimentLayer{DISCRETE}(prob::DEProblem, alg::DEAlgorithm, shooting_points::AbstractVector{<:Real}, params::Vector{Vector{Int64}} = [eachindex(prob.p) for _ in 1:nexp]; measurements = [], observed = default_observed, kwargs...) where {DISCRETE}
    nexp = length(params)
    layers = map(params) do param
        OEDLayer{DISCRETE}(prob, alg, shooting_points...; params = param, measurements = measurements, observed = observed, kwargs...)
    end |> Tuple
    all_params = union(params...)
    common = sort(all_params)
    idxmap = Dict(val => i for (i, val) in enumerate(common))

    new_params = (; original = params, all = common, permutation = idxmap)

    return MultiExperimentLayer{DISCRETE, false, true, MultipleShootingLayer, typeof(layers), typeof(new_params)}(layers, nexp, new_params)
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, multi::MultiExperimentLayer{<:Any, <:Any, true})
    exp_names = Tuple([Symbol("experiment_$i") for i in 1:multi.n_exp])
    exp_ps = Tuple(
        map(1:multi.n_exp) do i
            LuxCore.initialparameters(rng, multi.layers[i])
        end
    )
    return NamedTuple{exp_names}(exp_ps)
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, multi::MultiExperimentLayer{<:Any, <:Any, false})
    exp_names = Tuple([Symbol("experiment_$i") for i in 1:multi.n_exp])
    exp_ps = Tuple([LuxCore.initialparameters(rng, multi.layers) for _ in 1:multi.n_exp])
    return NamedTuple{exp_names}(exp_ps)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, multi::MultiExperimentLayer{<:Any, <:Any, true})
    exp_names = Tuple([Symbol("experiment_$i") for i in 1:multi.n_exp])
    exp_ps = Tuple(
        map(1:multi.n_exp) do i
            LuxCore.initialstates(rng, multi.layers[i])
        end
    )
    return NamedTuple{exp_names}(exp_ps)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, multi::MultiExperimentLayer{<:Any, <:Any, false})
    exp_names = Tuple([Symbol("experiment_$i") for i in 1:multi.n_exp])
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
    sols = map(enumerate(layer.layers)) do (i, _layer)
        ps_local, st_local = getproperty(ps, Symbol("experiment_$i")), getproperty(st, Symbol("experiment_$i"))
        sol, _ = _layer(x, ps_local, st_local)
        sol
    end
    return sols, st
end

n_observed(layer::MultiExperimentLayer{<:Any, <:Any, false}) = layer.n_exp * length(layer.layers.sampling_indices)
n_observed(layer::MultiExperimentLayer{<:Any, <:Any, true}) = sum(map(x -> length(x.sampling_indices), layer.layers))
Corleone.get_number_of_shooting_constraints(multi::MultiExperimentLayer{<:Any, <:Any, false, <:MultipleShootingLayer}) = multi.n_exp * Corleone.get_number_of_shooting_constraints(multi.layers)
Corleone.get_number_of_shooting_constraints(multi::MultiExperimentLayer{<:Any, <:Any, true, <:MultipleShootingLayer}) = sum(map(Corleone.get_number_of_shooting_constraints, multi.layers))
Corleone.get_number_of_shooting_constraints(multi::MultiExperimentLayer{<:Any, <:Any, <:Any, <:SingleShootingLayer}) = 0


function get_sampling_sums(multi::MultiExperimentLayer{<:Any, <:Any, true}, x, ps, st::NamedTuple{fields}) where {fields}
    return reduce(
        vcat, map(enumerate(fields)) do (i, field)
            get_sampling_sums(multi.layers[i], x, getproperty(ps, field), getproperty(st, field))
        end
    )
end

function get_sampling_sums!(res::AbstractVector, multi::MultiExperimentLayer{<:Any, <:Any, true}, x, ps, st::NamedTuple{fields}) where {fields}
    n_obs = cumsum(vcat(0, [length(x.sampling_indices) for x in multi.layers]))
    for (i, field) in zip(1:length(multi.layers), fields)
        get_sampling_sums!(view(res, (n_obs[i] + 1):n_obs[i + 1]), multi.layers[i], x, getproperty(ps, field), getproperty(st, field))
    end
    return
end

function get_sampling_sums(multi::MultiExperimentLayer{<:Any, <:Any, false}, x, ps, st::NamedTuple{fields}) where {fields}
    return reduce(
        vcat, map(fields) do field
            get_sampling_sums(multi.layers, x, getproperty(ps, field), getproperty(st, field))
        end
    )
end

function get_sampling_sums!(res::AbstractVector, multi::MultiExperimentLayer{<:Any, <:Any, false}, x, ps, st::NamedTuple{fields}) where {fields}
    n_obs = length(multi.layers.sampling_indices)
    for (i, field) in enumerate(fields)
        get_sampling_sums!(view(res, ((i - 1) * n_obs + 1):(i * n_obs)), multi.layers, x, getproperty(ps, field), getproperty(st, field))
    end
    return
end

function __fisher_information(multi::MultiExperimentLayer{<:Any, true, false}, trajs::Vector{<:Trajectory}, ps, st::NamedTuple{fields}) where {fields}
    return sum(
        map(zip(trajs, fields)) do (traj, field)
            __fisher_information(multi.layers, traj, getproperty(ps, field), getproperty(st, field))
        end
    )
end

function __fisher_information(multi::MultiExperimentLayer{<:Any, true, true}, trajs::Vector{<:Trajectory}, ps, st::NamedTuple{fields}) where {fields}
    fims = map(zip(enumerate(trajs), fields)) do ((i, traj), field)
        __fisher_information(multi.layers[i], traj, getproperty(ps, field), getproperty(st, field))
    end

    np = length(multi.params.all)
    F = zeros(eltype(fims[1]), (np, np))

    for (i, fim) in enumerate(fims)
        idxs = [multi.params.permutation[j] for j in multi.params.original[i]]
        F[idxs, idxs] .+= fim
    end

    return F
end

function fisher_information(multi::MultiExperimentLayer{<:Any, <:Any, false}, x, ps, st::NamedTuple{fields}) where {fields}
    return sum(
            map(fields) do field
                fisher_information(multi.layers, x, getproperty(ps, field), getproperty(st, field))[1]
        end
        ), st
end

function fisher_information(multi::MultiExperimentLayer{<:Any, <:Any, true}, x, ps, st::NamedTuple{fields}) where {fields}
    fim = map(enumerate(fields)) do (i, field)
        fisher_information(multi.layers[i], x, getproperty(ps, field), getproperty(st, field))[1]
    end
    np = length(multi.params.all)
    F = zeros(eltype(fim[1]), (np, np))
    for (i, fimi) in enumerate(fim)
        idxs = [multi.params.permutation[j] for j in multi.params.original[i]]
        F[idxs, idxs] .+= fimi
    end

    return F, st
end

Corleone.get_bounds(layer::MultiExperimentLayer{<:Any, <:Any, true}) = begin
    exp_names = Tuple([Symbol("experiment_$i") for i in 1:layer.n_exp])
    exp_bounds = map(Tuple(1:layer.n_exp)) do i
        Corleone.get_bounds(layer.layers[i])
    end
    NamedTuple{exp_names}(first.(exp_bounds)), NamedTuple{exp_names}(last.(exp_bounds))
end

Corleone.get_bounds(layer::MultiExperimentLayer{<:Any, <:Any, false}) = begin
    exp_names = Tuple([Symbol("experiment_$i") for i in 1:layer.n_exp])
    exp_bounds = map(Tuple(1:layer.n_exp)) do i
        Corleone.get_bounds(layer.layers)
    end
    NamedTuple{exp_names}(first.(exp_bounds)), NamedTuple{exp_names}(last.(exp_bounds))
end

"""
$(SIGNATURES)

Computes the block structure as defined by the `MultiExperimentLayer`, which may come from
two levels: 1) the different experiments, and 2) multiple shooting discretizations on the
experiment level.
"""
function Corleone.get_block_structure(layer::MultiExperimentLayer{<:Any, <:Any, true})
    blocks = map(layer.layers) do _layer
        Corleone.get_block_structure(_layer)
    end |> Tuple

    for i in 1:(layer.n_exp - 1)
        blocks[i + 1] .= blocks[i + 1] .+ blocks[i][end]
    end
    block_structure = reduce(vcat, [i == 1 ? blocks[i] : blocks[i][2:end] for i in 1:layer.n_exp])

    return block_structure
end

function Corleone.get_block_structure(layer::MultiExperimentLayer{<:Any, <:Any, false})
    blocks = map(1:layer.n_exp) do i
        Corleone.get_block_structure(layer.layers)
    end |> Tuple

    for i in 1:(layer.n_exp - 1)
        blocks[i + 1] .= blocks[i + 1] .+ blocks[i][end]
    end
    block_structure = reduce(vcat, [i == 1 ? blocks[i] : blocks[i][2:end] for i in 1:layer.n_exp])

    return block_structure
end

function Corleone.shooting_constraints(trajs::AbstractVector{<:Trajectory})
    return reduce(vcat, reduce(vcat, shooting_violations.(trajs)))
end

function Corleone.shooting_constraints!(res::AbstractVector, trajs::AbstractVector{<:Trajectory})
    i = 0
    for traj in trajs
        for subvec in traj.shooting, j in eachindex(subvec)
            i += 1
            res[i] = subvec[j]
        end
    end
    return res
end
