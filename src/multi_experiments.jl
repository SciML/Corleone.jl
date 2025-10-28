"""
$(TYPEDEF)
Generalization of OEDLayer to multiple experiments that can be jointly optimized.
# Fields
$(FIELDS)
"""
struct MultiExperimentLayer{fixed, L} <: LuxCore.AbstractLuxLayer
    "Layers defining multiexperiments"
    layers::L
    "Number of experiments"
    n_exp::Int
end

"""
    MultiExperimentLayer(oedlayer, n_exp)
Constructs a MultiExperimentLayer with `n_exp` experiments from a single `OEDLayer`, i.e.,
all experiments have the same degrees of freedom specified in `oedlayer`.
"""
function MultiExperimentLayer(oedlayer::OEDLayer, n_exp::Int)
    fixed = is_fixed(oedlayer)
    MultiExperimentLayer{fixed, OEDLayer}(oedlayer, n_exp)
end

"""
    MultiExperimentLayer(oedlayers...)
Constructs a MultiExperimentLayer several `OEDLayer`, i.e., different experiments may have
different degrees of freedom specified in their respective `OEDLayer`.
"""
function MultiExperimentLayer(layers::OEDLayer...)
    fixed = all(is_fixed.(layers))
    n = length(layers)
    MultiExperimentLayer{fixed, typeof(layers)}(layers, n)
end

function (layer::MultiExperimentLayer{<:Any, OEDLayer})(::Any, ps, st)
    sols = map(1:layer.n_exp) do i
        ps_local, st_local = getproperty(ps, Symbol("experiment_$i")), getproperty(st, Symbol("experiment_$i"))
        sol, _ = layer.layers(nothing, ps_local, st_local)
        sol
    end
    return sols, st
end

function (layer::MultiExperimentLayer{<:Any, <:Tuple})(::Any, ps, st)
    sols = map(1:layer.n_exp) do i
        ps_local, st_local = getproperty(ps, Symbol("experiment_$i")), getproperty(st, Symbol("experiment_$i"))
        sol, _ = layer.layers[i](nothing, ps_local, st_local)
        sol
    end
    return sols, st
end

(crit::AbstractCriterion)(multilayer::MultiExperimentLayer{<:Any, OEDLayer}) = begin
    ps, st = LuxCore.setup(Random.default_rng(), multilayer)

    pred_fim = fim(multilayer.layers)

    (p, ::Any) -> let ax = getaxes(ComponentArray(ps)), nexp = multilayer.n_exp, pred_fim=pred_fim
        ps = ComponentArray(p, ax)
        F = sum(map(1:nexp) do i
            local_p = getproperty(ps, Symbol("experiment_$i"))
            pred_fim(local_p, nothing)
        end)
        crit(F)
    end
end

(crit::AbstractCriterion)(multilayer::MultiExperimentLayer{<:Any, <:Tuple}) = begin
    ps, _ = LuxCore.setup(Random.default_rng(), multilayer)
    fims = map(multilayer.layers) do layer
        fim(layer)
    end

    (p, ::Any) -> let ax = getaxes(ComponentArray(ps)), fims=fims
        _p = ComponentArray(p, ax)
        F = sum(map(enumerate(fims)) do (i,local_fim)
            local_p = getproperty(_p, Symbol("experiment_$i"))
            local_fim(local_p, nothing)
        end)
        return crit(F)
    end
end

function fim(multilayer::MultiExperimentLayer{<:Any, OEDLayer}, p::AbstractArray)
    ps, st = LuxCore.setup(Random.default_rng(), multilayer)
    _p = ComponentArray(p, getaxes(ComponentArray(ps)))
    sum(map(1:multilayer.n_exp) do i
        local_p = getproperty(_p, Symbol("experiment_$i"))
        fim(multilayer.layers, local_p)
    end)
end

function fim(multilayer::MultiExperimentLayer{<:Any, <:Tuple}, p::AbstractArray)
    ps, st = LuxCore.setup(Random.default_rng(), multilayer)
    _p = ComponentArray(p, getaxes(ComponentArray(ps)))
    sum(map(1:multilayer.n_exp) do i
        local_p = getproperty(_p, Symbol("experiment_$i"))
        fim(multilayer.layers[i], local_p)
    end)
end


function LuxCore.initialparameters(rng::Random.AbstractRNG, multiexp::MultiExperimentLayer)
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multiexp.n_exp])
    exp_ps    = begin
        if isa(multiexp.layers, OEDLayer)
            Tuple([LuxCore.initialparameters(rng, multiexp.layers) for _ in 1:multiexp.n_exp])
        else
            Tuple([LuxCore.initialparameters(rng, multiexp.layers[i]) for i in 1:multiexp.n_exp])
        end
    end
    NamedTuple{exp_names}(exp_ps)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, multiexp::MultiExperimentLayer)
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multiexp.n_exp])
    exp_st    = begin
        if isa(multiexp.layers, OEDLayer)
            Tuple([LuxCore.initialstates(rng, multiexp.layers) for _ in 1:multiexp.n_exp])
        else
            Tuple([LuxCore.initialstates(rng, multiexp.layers[i]) for i in 1:multiexp.n_exp])
        end
    end
    NamedTuple{exp_names}(exp_st)
end

get_bounds(layer::MultiExperimentLayer) = begin
    exp_names = Tuple([Symbol("experiment_$i") for i=1:layer.n_exp])
    exp_bounds = begin
        if isa(layer.layers, OEDLayer)
            map(Tuple(1:layer.n_exp)) do _
                get_bounds(layer.layers)
            end
        else
            map(Tuple(1:layer.n_exp)) do i
                get_bounds(layer.layers[i])
            end
        end
    end
    ComponentArray(NamedTuple{exp_names}(first.(exp_bounds))), ComponentArray(NamedTuple{exp_names}(last.(exp_bounds)))
end

"""
$(METHODLIST)
Returns the function to evaluate shooting constraints for the `MultiExperimentLayer`.
"""
function get_shooting_constraints(layer::MultiExperimentLayer)
    @assert typeof(layer.layers.layer) <: MultipleShootingLayer "Shooting constraints are only available for MultipleShootingLayer."
    shooting_contraints = get_shooting_constraints(layer.layers)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    ax = getaxes(ComponentArray(ps))
    matching = let ax=ax
        (sols, p) -> begin
            _p = isa(p, Array) ? ComponentArray(p, ax) : p
            reduce(vcat, map(1:layer.n_exp) do i
                shooting_contraints(sols[i], getproperty(_p, Symbol("experiment_$i")))
            end)
        end
    end
    return matching
end

function get_sampling_constraint(layer::MultiExperimentLayer{<:Any, OEDLayer})
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    p = ComponentArray(ps)
    sampling = get_sampling_constraint(layer.layers)

    sampling_cons = let ax = getaxes(p), sampling=sampling
        (p, ::Any) -> begin
            ps = ComponentArray(p, ax)
            reduce(vcat, map(1:layer.n_exp) do i
                sampling(ps["experiment_$i"], nothing)
            end)
        end
    end

    return sampling_cons
end

function get_sampling_constraint(layer::MultiExperimentLayer{<:Any, <:Tuple})
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    p = ComponentArray(ps)
    sampling = get_sampling_constraint.(layer.layers)


    sampling_cons = let ax = getaxes(p), sampling=sampling
        (p, ::Any) -> begin
            ps = ComponentArray(p, ax)
            reduce(vcat, map(1:layer.n_exp) do i
                sampling[i](ps["experiment_$i"], nothing)
            end)
        end
    end

    return sampling_cons
end

"""
$(METHODLIST)
Computes the block structure as defined by the `MultiExperimentLayer`, which may come from
two levels: 1) the different experiments, and 2) multiple shooting discretizations on the
experiment level.
"""
function get_block_structure(layer::MultiExperimentLayer)
    blocks = begin
        if isa(layer.layers, Tuple)
            map(layer.layers) do _layer
                get_block_structure(_layer)
            end
        else
            map(1:layer.n_exp) do i
                get_block_structure(layer.layers)
            end |> Tuple
        end
    end

    for i=1:layer.n_exp-1
        blocks[i+1] .= blocks[i+1] .+ blocks[i][end]
    end
    block_structure = reduce(vcat, [i == 1 ? blocks[i] : blocks[i][2:end] for i =1:layer.n_exp])

    return block_structure
end

"""
$(METHODLIST)
Initializes all multiple shooting node variables of the different experiments defined in
the `MultiExperimentLayer` according to the used initialization `f`.
"""
function (f::AbstractNodeInitialization)(rng::Random.AbstractRNG, layer::MultiExperimentLayer;
    params=LuxCore.setup(rng, layer),
    shooting_variables= isa(layer.layers, Tuple) ? eachindex(first(layer.layers).problem.u0) : eachindex(get_problem(layer.layers.layer).u0),
    kwargs...)

    ps, st = LuxCore.setup(rng, layer)

    exp_names = Tuple([Symbol("experiment_$i") for i=1:layer.n_exp])
    ps_init = begin
        if isa(layer.layers, Tuple)
            map(enumerate(layer.layers)) do (i,_layer)
                local_ps, local_st = getproperty(first(params), Symbol("experiment_$i")), getproperty(last(params), Symbol("experiment_$i"))
                f(rng, _layer; params=(local_ps, local_st), shooting_variables=shooting_variables, kwargs...)[1]
            end
        else
            map(1:layer.n_exp) do i
                local_ps, local_st = getproperty(first(params), Symbol("experiment_$i")), getproperty(last(params), Symbol("experiment_$i"))
                f(rng, layer.layers; params=(local_ps, local_st), shooting_variables=shooting_variables, kwargs...)[1]
            end
        end
    end
    return NamedTuple{exp_names}(ps_init), st
end
