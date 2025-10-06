

struct MultiExperimentLayer{fixed, L} <: LuxCore.AbstractLuxLayer
    layers::L
    n_exp::Int
end

function MultiExperimentLayer(layer::OEDLayer, n_exp::Int)
    fixed = is_fixed(layer)
    MultiExperimentLayer{fixed, OEDLayer}(layer, n_exp)
end

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

(crit::AbstractCriterion)(multiexp::MultiExperimentLayer, sols::AbstractVector{<:DiffEqArray}) = begin
    fsym = Corleone.fisher_variables(multiexp.layers.layer)
    sumF = sum(map(sols) do sol
        Fi = sol[fsym][end]
        Fi
    end)
    crit(Corleone.symmetric_from_vector(sumF))
end

(crit::AbstractCriterion)(multiexp::MultiExperimentLayer, sols::AbstractVector{<:EnsembleSolution}) = begin
    fsym = Corleone.fisher_variables(multiexp.layers.layer)
    sumF = sum(map(sols) do sol
        Fi = last(sol)[fsym][end]
        Fi
    end)
    crit(Corleone.symmetric_from_vector(sumF))
end

(crit::AbstractCriterion)(multilayer::MultiExperimentLayer{true, OEDLayer}) = begin
    ps, st = LuxCore.setup(Random.default_rng(), multilayer)
    sols, _ = multilayer(nothing, ps, st)
    nc = vcat(0, cumsum(map(x -> length(x.t), multilayer.layers.layer.controls))...)
    tinf = last(multilayer.layers.layer.problem.tspan)
    Fs = map(sols) do sol_i
            map(enumerate(multilayer.layers.layer.controls)) do (i,sampling) # All fixed -> only sampling controls
            Fi = sort(Corleone.observed_sensitivity_product_variables(multilayer.layers.layer, i), by= x -> split(string(x), "ˏ")[3])
            wts= vcat(sampling.t, tinf) |> unique!
            idxs = findall(x -> x in wts, sol_i.t)
            diff(sol_i[Fi][idxs])
        end
    end

    (p, ::Any) -> let Fs = Fs, ax = getaxes(ComponentArray(ps)), nc=nc
        ps = ComponentArray(p, ax)
        F = symmetric_from_vector(sum(map((enumerate(Fs))) do (i,Fi)
            local_p = getproperty(ps, Symbol("experiment_$i"))
            sum(map(zip(Fi, nc[1:end-1], nc[2:end])) do (F_hi, idx_start, idx_end)
                local_sampling = local_p.controls[idx_start+1:idx_end]
                sum(map(zip(F_hi, local_sampling)) do (F_it, wit)
                    F_it * wit
                end)
            end)
        end))
        crit(F)
    end
end

(crit::AbstractCriterion)(multilayer::MultiExperimentLayer{true, <:Tuple}) = begin
    ps, st = LuxCore.setup(Random.default_rng(), multilayer)
    sols, _ = multilayer(nothing, ps, st)
    nc = [vcat(0, cumsum(map(x -> length(x.t), layer.layer.controls))...) for layer in multilayer.layers]
    tinfs = [last(layer.layer.problem.tspan) for layer in multilayer.layers]
    Fs = map(enumerate(sols)) do (j,sol_i)
            map(enumerate(multilayer.layers[j].layer.controls)) do (i,sampling) # All fixed -> only sampling controls
            Fi = sort(Corleone.observed_sensitivity_product_variables(multilayer.layers[j].layer, i), by= x -> split(string(x), "ˏ")[3])
            wts= vcat(sampling.t, tinfs[j]) |> unique!
            idxs = findall(x -> x in wts, sol_i.t)
            diff(sol_i[Fi][idxs])
        end
    end

    (p, ::Any) -> let Fs = Fs, ax = getaxes(ComponentArray(ps)), nc=nc
        ps = ComponentArray(p, ax)
        F = symmetric_from_vector(sum(map((enumerate(Fs))) do (i,Fi)
            local_p = getproperty(ps, Symbol("experiment_$i"))
            sum(map(zip(Fi, nc[i][1:end-1], nc[i][2:end])) do (F_hi, idx_start, idx_end)
                local_sampling = local_p.controls[idx_start+1:idx_end]
                sum(map(zip(F_hi, local_sampling)) do (F_it, wit)
                    F_it * wit
                end)
            end)
        end))
        crit(F)
    end
end


(crit::AbstractCriterion)(multilayer::MultiExperimentLayer{false}) = begin
    ps, st = LuxCore.setup(Random.default_rng(), multilayer)
    (p, ::Any) -> let ax = getaxes(ComponentArray(ps)), st = st, layer=multilayer
        ps = ComponentArray(p, ax)
        sol, _ = layer(nothing, ps, st)
        crit(layer, sol)
    end
end


function LuxCore.initialparameters(rng::Random.AbstractRNG, multiexp::MultiExperimentLayer{<:Any, OEDLayer})
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multiexp.n_exp])
    exp_ps    = Tuple([LuxCore.initialparameters(rng, multiexp.layers) for _ in 1:multiexp.n_exp])
    NamedTuple{exp_names}(exp_ps)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, multiexp::MultiExperimentLayer{<:Any, OEDLayer})
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multiexp.n_exp])
    exp_st    = Tuple([LuxCore.initialstates(rng, multiexp.layers) for _ in 1:multiexp.n_exp])
    NamedTuple{exp_names}(exp_st)
end

function LuxCore.initialparameters(rng::Random.AbstractRNG, multiexp::MultiExperimentLayer{<:Any, <:Tuple})
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multiexp.n_exp])
    exp_ps    = Tuple([LuxCore.initialparameters(rng, multiexp.layers[i]) for i in 1:multiexp.n_exp])
    NamedTuple{exp_names}(exp_ps)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, multiexp::MultiExperimentLayer{<:Any, <:Tuple})
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multiexp.n_exp])
    exp_st    = Tuple([LuxCore.initialstates(rng, multiexp.layers[i]) for i in 1:multiexp.n_exp])
    NamedTuple{exp_names}(exp_st)
end

get_bounds(layer::MultiExperimentLayer{<:Any, OEDLayer}) = begin
    exp_names = Tuple([Symbol("experiment_$i") for i=1:layer.n_exp])
    exp_bounds = map(Tuple(1:layer.n_exp)) do _
        get_bounds(layer.layers)
    end
    ComponentArray(NamedTuple{exp_names}(first.(exp_bounds))), ComponentArray(NamedTuple{exp_names}(last.(exp_bounds)))
end

get_bounds(layer::MultiExperimentLayer{<:Any, <:Tuple}) = begin
    exp_names = Tuple([Symbol("experiment_$i") for i=1:layer.n_exp])
    exp_bounds = map(Tuple(1:layer.n_exp)) do i
        get_bounds(layer.layers[i])
    end
    ComponentArray(NamedTuple{exp_names}(first.(exp_bounds))), ComponentArray(NamedTuple{exp_names}(last.(exp_bounds)))
end

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
