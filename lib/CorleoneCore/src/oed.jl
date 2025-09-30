struct OEDLayer{fixed,L,O,D} <: LuxCore.AbstractLuxLayer
    layer::L
    observed::O
    dimensions::D
end

function OEDLayer(prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.AbstractDEAlgorithm;
            control_indices = Int64[],
            controls = nothing,
            tunable_ic = Int64[],
            bounds_ic = nothing,
            observed = prob.f.observed == SciMLBase.DEFAULT_OBSERVED ? (u,p,t) -> u[eachindex(prob.u0)] : prob.f.observed,
            dt = (-)(reverse(prob.tspan)...)/100,
            params = setdiff(eachindex(prob.p), control_indices),
            kwargs...)

    layer = SingleShootingLayer(prob, alg; tunable_ic = tunable_ic, controls = controls,
                                control_indices = control_indices, bounds_ic=bounds_ic,
                                kwargs...)
    OEDLayer(layer; observed = observed, params = params, dt =dt)
end

function OEDLayer(prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.AbstractDEAlgorithm,
            shooting_points;
            control_indices = Int64[],
            controls = nothing,
            tunable_ic = Int64[],
            bounds_ic = nothing,
            bounds_nodes = nothing,
            observed = prob.f.observed == SciMLBase.DEFAULT_OBSERVED ? (u,p,t) -> u[eachindex(prob.u0)] : prob.f.observed,
            ensemble_alg=EnsembleSerial(),
            dt = (-)(reverse(prob.tspan)...)/100,
            params = setdiff(prob.p, control_indices),
            kwargs...)

    layer = MultipleShootingLayer(prob, alg, control_indices, controls, shooting_points;
                        tunable_ic=tunable_ic, bounds_ic=bounds_ic, bounds_nodes=bounds_nodes,
                        ensemble_alg=ensemble_alg, kwargs...)

    OEDLayer(layer; observed = observed, params = params, dt =dt)
end

function is_fixed(layer::Union{SingleShootingLayer, MultipleShootingLayer})
    controls, control_indices = get_controls(layer)
    isempty(get_tunable(layer)) && (isempty(control_indices) || isnothing(controls))
end

function is_fixed(layer::OEDLayer{true, <:Any, <:Any, <:Any})
    true
end

function is_fixed(layer::OEDLayer{false, <:Any, <:Any, <:Any})
    false
end

function OEDLayer(layer::Union{SingleShootingLayer,MultipleShootingLayer};
                    observed = (u,p,t) -> u,
                    params = get_params(layer),
                    dt = (-)(reverse(tspan)...)/100
                    )

    prob = get_problem(layer)
    _, control_indices = get_controls(layer)
    nx, np, nc, np_considered = length(prob.u0), length(prob.p), length(control_indices), length(params)

    fixed = is_fixed(layer)

    oed_layer = augment_layer_for_oed(layer, params=params, observed=observed, dt=dt)

    obs = begin
        if fixed
            x, p, t = Symbolics.variables(:x, 1:nx), Symbolics.variables(:p, 1:np), Symbolics.variable(:t)

            h = observed(x,p,t)
            hx = Symbolics.jacobian(h, x)
            hx_fun = Symbolics.build_function(hx, x, p, t, expression = Val{false})[1]

            (h = observed, hx = hx_fun)
        else
            nothing
        end
    end

    dimensions = (np = np, nh = length(observed(prob.u0, prob.p, first(prob.tspan))),
                  np_fisher = np_considered, nc = nc, nx = nx)
    return OEDLayer{fixed, typeof(oed_layer), typeof(obs), typeof(dimensions)}(oed_layer, obs, dimensions)
end

LuxCore.initialparameters(rng::Random.AbstractRNG, layer::OEDLayer) = LuxCore.initialparameters(rng, layer.layer)
LuxCore.initialstates(rng::Random.AbstractRNG, layer::OEDLayer) = LuxCore.initialstates(rng, layer.layer)

function (layer::OEDLayer)(::Any, ps, st)
    layer.layer(nothing, ps, st)
end

(crit::AbstractCriterion)(oedlayer::OEDLayer{true, <:SingleShootingLayer, <:Any, <:Any}) = begin
    ps, st = LuxCore.setup(Random.default_rng(), oedlayer)
    sols, _ = oedlayer(nothing, ps, st)
    nc = vcat(0, cumsum(map(x -> length(x.t), oedlayer.layer.controls))...)
    tinf = last(oedlayer.layer.problem.tspan)
    Fs = map(enumerate(oedlayer.layer.controls)) do (i,sampling) # All fixed -> only sampling controls
        Fi = sort(CorleoneCore.observed_sensitivity_product_variables(oedlayer.layer, i), by= x -> split(string(x), "ˏ")[3])
        wts= vcat(sampling.t, tinf) |> unique!
        idxs = findall(x -> x in wts, sols.t)
        diff(sols[Fi][idxs])
    end

    (p, ::Any) -> let Fs = Fs, ax = getaxes(ComponentArray(ps)), nc=nc
        ps = ComponentArray(p, ax)
        F = symmetric_from_vector(sum(map(zip(Fs, nc[1:end-1], nc[2:end])) do (F_i, idx_start, idx_end)
            local_sampling = ps.controls[idx_start+1:idx_end]
            sum(map(zip(F_i, local_sampling)) do (F_it, wit)
                F_it * wit
            end)
        end))
        crit(F)
    end
end

(crit::AbstractCriterion)(oedlayer::OEDLayer{false}) = begin
    ps, st = LuxCore.setup(Random.default_rng(), oedlayer)
    (p, ::Any) -> let ax = getaxes(ComponentArray(ps)), st = st, layer=oedlayer.layer
        ps = ComponentArray(p, ax)
        sol, _ = layer(nothing, ps, st)
        crit(layer, sol)
    end
end

function (init::AbstractNodeInitialization)(rng::AbstractRNG, layer::OEDLayer; kwargs...)
    init(rng, layer.layer; kwargs...)
end

get_bounds(layer::OEDLayer) = get_bounds(layer.layer)
get_shooting_constraints(layer::OEDLayer{false, <:MultipleShootingLayer, <:Any, <:Any}) = get_shooting_constraints(layer.layer)
get_block_structure(layer::OEDLayer) = get_block_structure(layer.layer)
sensitivity_variables(layer::OEDLayer) = sensitivity_variables(layer.layer)
fisher_variables(layer::OEDLayer) = fisher_variables(layer.layer)


struct MultiExperimentLayer{fixed} <: LuxCore.AbstractLuxLayer
    layer::OEDLayer
    n_exp::Int
end

function MultiExperimentLayer(layer::OEDLayer, n_exp::Int)
    fixed = is_fixed(layer)
    MultiExperimentLayer{fixed}(layer, n_exp)
end

function (layer::MultiExperimentLayer)(::Any, ps, st)
    sols = map(1:layer.n_exp) do i
        ps_local, st_local = getproperty(ps, Symbol("experiment_$i")), getproperty(st, Symbol("experiment_$i"))
        sol, _ = layer.layer(nothing, ps_local, st_local)
        sol
    end
    return sols, st
end

(crit::AbstractCriterion)(multiexp::MultiExperimentLayer, sols::AbstractVector{<:DiffEqArray}) = begin
    fsym = CorleoneCore.fisher_variables(multiexp.layer.layer)
    sumF = sum(map(sols) do sol
        Fi = sol[fsym][end]
        Fi
    end)
    crit(CorleoneCore.symmetric_from_vector(sumF))
end

(crit::AbstractCriterion)(multiexp::MultiExperimentLayer, sols::AbstractVector{<:EnsembleSolution}) = begin
    fsym = CorleoneCore.fisher_variables(multiexp.layer.layer)
    sumF = sum(map(sols) do sol
        Fi = last(sol)[fsym][end]
        Fi
    end)
    crit(CorleoneCore.symmetric_from_vector(sumF))
end

(crit::AbstractCriterion)(multilayer::MultiExperimentLayer{true}) = begin
    ps, st = LuxCore.setup(Random.default_rng(), multilayer)
    sols, _ = multilayer(nothing, ps, st)
    nc = vcat(0, cumsum(map(x -> length(x.t), multilayer.layer.layer.controls))...)
    tinf = last(multilayer.layer.layer.problem.tspan)
    Fs = map(sols) do sol_i
            map(enumerate(multilayer.layer.layer.controls)) do (i,sampling) # All fixed -> only sampling controls
            Fi = reshape(sort(CorleoneCore.observed_sensitivity_product_variables(multilayer.layer.layer, i), by= x -> split(string(x), "ˏ")[3]), (multilayer.layer.dimensions.np_fisher,multilayer.layer.dimensions.np_fisher))
            wts= vcat(sampling.t, tinf) |> unique!
            idxs = findall(x -> x in wts, sol_i.t)
            diff(sol_i[Fi][idxs])
        end
    end

    (p, ::Any) -> let Fs = Fs, ax = getaxes(ComponentArray(ps)), nc=nc
        ps = ComponentArray(p, ax)
        F = Symmetric(sum(map((enumerate(Fs))) do (i,Fi)
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


(crit::AbstractCriterion)(multilayer::MultiExperimentLayer{false}) = begin
    ps, st = LuxCore.setup(Random.default_rng(), multilayer)
    (p, ::Any) -> let ax = getaxes(ComponentArray(ps)), st = st, layer=multilayer
        ps = ComponentArray(p, ax)
        sol, _ = layer(nothing, ps, st)
        crit(layer, sol)
    end
end


function LuxCore.initialparameters(rng::Random.AbstractRNG, multiexp::MultiExperimentLayer)
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multiexp.n_exp])
    exp_ps    = Tuple([LuxCore.initialparameters(rng, multiexp.layer) for _ in 1:multiexp.n_exp])
    NamedTuple{exp_names}(exp_ps)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, multiexp::MultiExperimentLayer)
    exp_names = Tuple([Symbol("experiment_$i") for i=1:multiexp.n_exp])
    exp_st    = Tuple([LuxCore.initialstates(rng, multiexp.layer) for _ in 1:multiexp.n_exp])
    NamedTuple{exp_names}(exp_st)
end

get_bounds(layer::MultiExperimentLayer) = begin
    exp_names = Tuple([Symbol("experiment_$i") for i=1:layer.n_exp])
    exp_bounds = map(Tuple(1:layer.n_exp)) do _
        get_bounds(layer.layer)
    end
    ComponentArray(NamedTuple{exp_names}(first.(exp_bounds))), ComponentArray(NamedTuple{exp_names}(last.(exp_bounds)))
end

function get_shooting_constraints(layer::MultiExperimentLayer)
    @assert typeof(layer.layer.layer) <: MultipleShootingLayer "Shooting constraints are only available for MultipleShootingLayer."
    shooting_contraints = get_shooting_constraints(layer.layer)
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