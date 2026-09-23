"""
$(TYPEDEF)

Generalization of OEDLayer to multiple experiments that can be jointly optimized.

# Fields
$(FIELDS)
"""
struct MultiExperimentLayer{SPLIT, L, P} <: LuxCore.AbstractLuxContainerLayer{(:experiments,)}
    "The different specifications of the experiments in OEDLayers, perhaps different"
    experiments::L
    "Parameter metadata"
    params::P
    "Number of experiments considered"
    n_exp::Int
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

"""
$(SIGNATURES)

Constructs a multi-experiment OED layer from one differential equation problem.

# Arguments
- `prob`: Differential equation problem shared by the experiments.
- `alg`: Differential equation solver algorithm.
- `nexp`: Number of experiments, or pass a vector of parameter-index vectors to split
  parameters across experiments.

# Keywords
- `params`: Parameter indices included in each experiment.
- `measurements`: Optional measurement controls.

# Returns
A `MultiExperimentLayer` whose parameters and states are grouped by experiment.
"""
function MultiExperimentLayer(
        prob::SciMLBase.AbstractDEProblem,
        variable_id,
        nexp::Int,
        controls...;
        params = eachindex(prob.p), 
        measurements = AbstractMeasurement[],
        algorithm::SciMLBase.AbstractDEAlgorithm, 
        kwargs...
    )
    
    layer = OEDLayer(prob, variable_id, params, controls...; algorithm = algorithm, 
        measurements = measurements, 
        kwargs...
    )
    return MultiExperimentLayer{false, typeof(layer), typeof(params)}(layer, params, nexp)
end

function MultiExperimentLayer(
        prob::SciMLBase.AbstractDEProblem,
        variable_id,
        params::Vector{<:Vector{<:Int}},
        controls...;
        algorithm::SciMLBase.AbstractDEAlgorithm, 
        measurements = AbstractMeasurement[],
        kwargs...)

    nexp = length(params)
    layers = map(params) do param
        OEDLayer(prob, variable_id, param, controls...; 
            algorithm = algorithm,
            measurements = measurements,
            kwargs...)
    end |> Tuple

    all_params = union(params...)
    common = sort(all_params)
    idxmap = Dict(val => i for (i, val) in enumerate(common))

    new_params = (; original = params, all = common, permutation = idxmap)

    return MultiExperimentLayer{true, typeof(layers), typeof(new_params)}(layers, new_params, nexp)
end


function LuxCore.initialparameters(rng::Random.AbstractRNG, multi::MultiExperimentLayer{true})
    exp_names = Tuple([Symbol("experiment_$i") for i in 1:multi.n_exp])
    exp_ps = Tuple(
        map(1:multi.n_exp) do i
            LuxCore.initialparameters(rng, multi.experiments[i])
        end
    )
    return NamedTuple{exp_names}(exp_ps)
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, multi::MultiExperimentLayer{false})
    exp_names = Tuple([Symbol("experiment_$i") for i in 1:multi.n_exp])
    exp_ps = Tuple([LuxCore.initialparameters(rng, multi.experiments) for i in 1:multi.n_exp])
    return NamedTuple{exp_names}(exp_ps)
end


function LuxCore.initialstates(rng::Random.AbstractRNG, multi::MultiExperimentLayer{true})
    exp_names = Tuple([Symbol("experiment_$i") for i in 1:multi.n_exp])
    exp_ps = Tuple(
        map(1:multi.n_exp) do i
            LuxCore.initialstates(rng, multi.experiments[i])
        end
    )
    np = length(multi.params.all)

    st1 = exp_ps[1]
    F_init = zeros(eltype(exp_ps[1].F_init), np, np)
    st1 = merge(st1, (; F_init = F_init))

    new_sts = (st1, exp_ps[2:end]...)

    return NamedTuple{exp_names}(new_sts)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, multi::MultiExperimentLayer{false})
    exp_names = Tuple([Symbol("experiment_$i") for i in 1:multi.n_exp])
    exp_ps = Tuple([LuxCore.initialstates(rng, multi.experiments) for i in 1:multi.n_exp])
    return NamedTuple{exp_names}(exp_ps)
end


function (multi::MultiExperimentLayer{false})(x, ps, st)
    sols = map(1:multi.n_exp) do i
        ps_local, st_local = getproperty(ps, Symbol("experiment_$i")), getproperty(st, Symbol("experiment_$i"))
        sol, _ = multi.experiments(x, ps_local, st_local)
        sol
    end
    return sols, st
end

function (multi::MultiExperimentLayer{true})(x, ps, st)
    sols = map(enumerate(multi.experiments)) do (i, layer)
        ps_local, st_local = getproperty(ps, Symbol("experiment_$i")), getproperty(st, Symbol("experiment_$i"))
        sol, _ = layer(x, ps_local, st_local)
        sol
    end
    return sols, st
end

function fisher_information(multi::MultiExperimentLayer{false}, x, ps, st::NamedTuple{fields}) where {fields}
    F = sum(map(fields) do field
        __fisher_information(multi.experiments, x, getproperty(ps, field), getproperty(st, field))
    end)
    
    F_init = st[1].F_init
    return F + F_init, st
end
#=
function fisher_information(multi::MultiExperimentLayer{true}, x, ps, st::NamedTuple{fields}) where {fields}

    fims = map(enumerate(fields)) do (i,field)
        __fisher_information(multi.experiments[i], x, getproperty(ps, field), getproperty(st, field))
    end

    np = length(multi.params.all)
    F = zeros(eltype(fims[1]), (np, np))

    for (i, fim) in enumerate(fims)
        idxs = [multi.params.permutation[j] for j in multi.params.original[i]]
        F[idxs, idxs] .+= fim
    end

    return F + st[1].F_init, st
end
=#

function fisher_information(multi::MultiExperimentLayer{true}, x, ps, st::NamedTuple{fields}) where {fields}
    np = length(multi.params.all)

    F = sum(eachindex(fields)) do i
        field = fields[i]
        fim_local = __fisher_information(
            multi.experiments[i],
            x,
            getproperty(ps, field),
            getproperty(st, field)
        )
        global_idxs = [multi.params.permutation[j] for j in multi.params.original[i]]

        full_fim = zeros(eltype(fim_local), np, np)
        full_fim[global_idxs, global_idxs] = fim_local
        full_fim
    end
                                                                                                                                                                                                                                 
    return F + st[1].F_init, st
end  

function sampling_sums(multi::MultiExperimentLayer{true}, x, ps, st::NamedTuple{fields}) where {fields}
    return reduce(
        vcat, map(enumerate(fields)) do (i, field)
            sampling_sums(multi.experiments[i], x, getproperty(ps, field), getproperty(st, field))
        end
    )
end

function sampling_sums(multi::MultiExperimentLayer{false}, x, ps, st::NamedTuple{fields}) where {fields}
    return reduce(
        vcat, map(enumerate(fields)) do (i, field)
            sampling_sums(multi.experiments, x, getproperty(ps, field), getproperty(st, field))
        end
    )
end

function sampling_sums!(res::AbstractVector, multi::MultiExperimentLayer{false}, x, ps, st::NamedTuple{fields}) where {fields}
    n_obs = size(multi.experiments.measurements.observed.local_information_gain.getters, 1)
    for (i, field) in enumerate(fields)
        sampling_sums!(view(res, ((i - 1) * n_obs + 1):(i * n_obs)), multi.experiments, x, getproperty(ps, field), getproperty(st, field))
    end
    return
end

function sampling_sums!(res::AbstractVector, multi::MultiExperimentLayer{true}, x, ps, st::NamedTuple{fields}) where {fields}
    current_start = 0
    for (i, field) in enumerate(fields)
        n_obs = size(multi.experiments[i].measurements.observed.local_information_gain.getters, 1)
        sampling_sums!(view(res, (current_start+1):(current_start+n_obs)), multi.experiments[i], x, getproperty(ps, field), getproperty(st, field))
        current_start += n_obs
    end
    return
end

Corleone.get_number_of_shooting_constraints(multi::MultiExperimentLayer{false}) = multi.n_exp * Corleone.get_number_of_shooting_constraints(multi.experiments)
Corleone.get_number_of_shooting_constraints(multi::MultiExperimentLayer{true}) = sum(map(Corleone.get_number_of_shooting_constraints, multi.experiments))
n_observed(layer::MultiExperimentLayer{false}) = layer.n_exp * n_observed(layer.experiments)
n_observed(layer::MultiExperimentLayer{true}) = sum(map(n_observed, layer.experiments))

#=

function update_fim(oed::MultiExperimentLayer{DISCRETE, FIXED, <:Any, <:SingleShootingLayer}, experiments, st::NamedTuple) where {DISCRETE, FIXED}
    FIM = sum(
        map(experiments) do experiment
            fisher_information(oed, nothing, experiment.ps, experiment.st)[1]
        end
    )

    st1 = getproperty(st, Symbol("experiment_1"))
    st1 = merge(st1, (; F_init = FIM + st[1].F_init))

    return merge(st, (; experiment_1 = st1))
end


function update_fim(oed::MultiExperimentLayer{DISCRETE, FIXED, <:Any, <:MultipleShootingLayer}, experiments, st::NamedTuple) where {DISCRETE, FIXED}
    FIM = sum(
        map(experiments) do experiment
            fisher_information(oed, nothing, experiment.ps, experiment.st)[1]
        end
    )

    st1 = getproperty(st, Symbol("experiment_1"))
    int1 = merge(st1.interval_1, (; F_init = FIM + st[1][1].F_init))
    st1 = merge(st1, (; interval_1 = int1))

    return merge(st, (; experiment_1 = st1))
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



function fisher_information(multi::MultiExperimentLayer{<:Any, <:Any, true}, x, ps, st::NamedTuple{fields}; add_initial = true) where {fields}
    fim = map(enumerate(fields)) do (i, field)
        fisher_information(multi.layers[i], x, getproperty(ps, field), getproperty(st, field); add_initial = false)[1]
    end
    np = length(multi.params.all)
    F = zeros(eltype(fim[1]), (np, np))
    for (i, fimi) in enumerate(fim)
        idxs = [multi.params.permutation[j] for j in multi.params.original[i]]
        F[idxs, idxs] .+= fimi
    end
    F_init = isa(multi.layers[1].layer, MultipleShootingLayer) ? st[1][1].F_init : st[1].F_init
    add_initial && return F + F_init, st
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

=#