"""
$(TYPEDEF)

Joint experimental design with independent Lux parameter/state trees per experiment.
Local information matrices are embedded in common physical-parameter index order
before summation. A global `st.F_init` is included once. SVD-reduced experiments
contribute their information projected back through their fixed direction bases.

Construct from a tuple/NamedTuple of OED layers, or `MultiExperimentLayer(oed, n)`.
Different experiments must use the same meaning for each physical parameter index.
Tuple inputs get names `experiment_1`, `experiment_2`, …; NamedTuples retain their
names. `n` must be positive and the experiment collection must be nonempty.

Evaluation returns `(named_trajectories, next_st)`. Parameters and state have named
`experiments` subtrees; access them by `getproperty`. Iterating ComponentArrays
instead yields numerical entries. Setup initializes independent numerical trees
for repeated experiments, while the layer descriptions may be shared. Parameter
buffers remain vector-based even though information is matrix-valued.

The global prior has size `length(params) × length(params)`, in sorted union index
order. Child priors are excluded from joint information. SVD embeddings project
local reduced information into physical coordinates as an approximation; they do
not recover discarded directions. See [multiple experiments](@ref oed_experiments).

# Fields
$(FIELDS)
"""
struct MultiExperimentLayer{E, P, B} <: LuxCore.AbstractLuxContainerLayer{(:experiments,)}
    "The OED layers, keyed by experiment name."
    experiments::E
    "The union of physical parameter indices in information-matrix order."
    params::P
    "Fixed embeddings of local sensitivity directions into the common parameter axes."
    embeddings::B
end

function MultiExperimentLayer(experiments::NamedTuple)
    !isempty(experiments) && all(layer -> layer isa OEDLayer, values(experiments)) ||
        throw(ArgumentError("Supply at least one OED layer."))
    params = sort!(unique!(reduce(vcat, map(layer -> layer.params, values(experiments)))))
    embeddings = map(values(experiments)) do layer
        [
            index in layer.params ? layer.basis[findfirst(==(index), layer.params), j] : zero(eltype(layer.basis))
                for index in params, j in axes(layer.basis, 2)
        ]
    end
    return MultiExperimentLayer(experiments, params, embeddings)
end
function MultiExperimentLayer(experiments::Tuple)
    names = ntuple(i -> Symbol(:experiment_, i), length(experiments))
    return MultiExperimentLayer(NamedTuple{names}(experiments))
end
function MultiExperimentLayer(layer::OEDLayer, n::Integer)
    n > 0 || throw(ArgumentError("The number of experiments must be positive."))
    return MultiExperimentLayer(ntuple(_ -> layer, n))
end

function LuxCore.initialstates(rng::Random.AbstractRNG, multi::MultiExperimentLayer)
    experiments = map(layer -> LuxCore.initialstates(rng, layer), multi.experiments)
    T = eltype(first(values(multi.experiments)).augmented_prob.u0)
    return (; experiments, F_init = zeros(T, length(multi.params), length(multi.params)))
end
LuxCore.statelength(multi::MultiExperimentLayer) = sum(LuxCore.statelength, values(multi.experiments)) + length(multi.params)^2

# Parameter containers need only provide named access: iterating a ComponentArray
# yields scalar storage, not its experiment subtrees.
function apply_experiment(operation, multi, ps, st, ::Val{name}) where {name}
    return operation(
        getproperty(multi.experiments, name),
        getproperty(ps.experiments, name), getproperty(st.experiments, name)
    )
end
function map_experiments(operation, multi, ps, st)
    return map(Val.(keys(multi.experiments))) do name
        apply_experiment(operation, multi, ps, st, name)
    end
end

function (multi::MultiExperimentLayer)(x, ps, st)
    results = map_experiments((layer, parameters, state) -> layer(x, parameters, state), multi, ps, st)
    names = keys(multi.experiments)
    trajectories = NamedTuple{names}(map(first, results))
    states = NamedTuple{names}(map(last, results))
    return trajectories, merge(st, (; experiments = states))
end

function experiment_information(multi, trajectories)
    blocks = map(values(multi.experiments), values(trajectories), multi.embeddings) do layer, trajectory, embedding
        F = fisher_information(layer, trajectory; add_initial = false)
        embedding * F * embedding'
    end
    return sum(blocks)
end
function fisher_information(multi::MultiExperimentLayer, x, ps, st; add_initial = true)
    trajectories, next_st = multi(x, ps, st)
    F = experiment_information(multi, trajectories)
    return add_initial ? F + st.F_init : F, next_st
end

sampling_sums(multi::MultiExperimentLayer, trajectories::NamedTuple) = reduce(vcat, values(map(sampling_sums, multi.experiments, trajectories)))
function sampling_sums(multi::MultiExperimentLayer, x, ps, st)
    return reduce(
        vcat, map_experiments(
            (layer, parameters, state) -> sampling_sums(layer, x, parameters, state), multi, ps, st
        )
    )
end
function sampling_sums!(res, multi::MultiExperimentLayer, args...)
    length(res) >= n_observed(multi) || throw(DimensionMismatch("Incorrect sampling budget buffer length."))
    copyto!(res, sampling_sums(multi, args...))
    return res
end
n_observed(multi::MultiExperimentLayer) = sum(n_observed, values(multi.experiments))
Corleone.get_number_of_shooting_constraints(multi::MultiExperimentLayer, ps, st) = sum(map_experiments(get_number_of_shooting_constraints, multi, ps, st))
Corleone.shooting_constraints(multi::MultiExperimentLayer, trajectories::NamedTuple) = reduce(vcat, values(map(shooting_constraints, trajectories)))
function Corleone.shooting_constraints!(res, multi::MultiExperimentLayer, trajectories::NamedTuple)
    copyto!(res, shooting_constraints(multi, trajectories))
    return res
end
function update_fim(multi::MultiExperimentLayer, experiments, st)
    isempty(experiments) && return st
    added = sum(experiments) do experiment
        local_st = hasproperty(experiment, :st) ? experiment.st : st
        first(fisher_information(multi, nothing, experiment.ps, local_st; add_initial = false))
    end
    return merge(st, (; F_init = st.F_init + added))
end
update_fim(multi::MultiExperimentLayer, experiments) = update_fim(multi, experiments, LuxCore.initialstates(Random.default_rng(), multi))

Corleone.optimization_schedule(::MultiExperimentLayer, st) = map(state -> state.timestops, st.experiments)
Corleone.optimization_matching(layer::MultiExperimentLayer, trajectories::NamedTuple) = shooting_constraints(layer, trajectories)
Corleone.optimization_matching!(res, layer::MultiExperimentLayer, trajectories::NamedTuple) = shooting_constraints!(res, layer, trajectories)

function sampling_integrality!(mask, layer::MultiExperimentLayer)
    for name in keys(layer.experiments)
        sampling_integrality!(getproperty(mask.experiments, name), layer.experiments[name])
    end
    return mask
end
