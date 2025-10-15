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
        x, p, t = Symbolics.variables(:x, 1:nx), Symbolics.variables(:p, 1:np), Symbolics.variable(:t)

        h = observed(x,p,t)
        hx = Symbolics.jacobian(h, x)
        hx_fun = Symbolics.build_function(hx, x, p, t, expression = Val{false}, cse=true)[1]

        (h = observed, hx = hx_fun)
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
        Fi = observed_sensitivity_product_variables(oedlayer.layer, i)
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
observed_sensitivity_product_variables(layer::OEDLayer, observed_idx::Int) = observed_sensitivity_product_variables(layer.layer, observed_idx)

function fim(oedlayer::OEDLayer{true, <:Any, <:Any, <:Any}, p::AbstractArray)
    ps, st = LuxCore.setup(Random.default_rng(), oedlayer)
    sols, _ = oedlayer(nothing, ps, st)
    nc = vcat(0, cumsum(map(x -> length(x.t), oedlayer.layer.controls))...)
    tinf = last(oedlayer.layer.problem.tspan)
    Fs = map(enumerate(oedlayer.layer.controls)) do (i,sampling) # All fixed -> only sampling controls
        Fi = observed_sensitivity_product_variables(oedlayer.layer, i)
        wts= vcat(sampling.t, tinf) |> unique!
        idxs = findall(x -> x in wts, sols.t)
        diff(sols[Fi][idxs])
    end

    ax = getaxes(ComponentArray(ps))
    ps = ComponentArray(p, ax)
    F = symmetric_from_vector(sum(map(zip(Fs, nc[1:end-1], nc[2:end])) do (F_i, idx_start, idx_end)
        local_sampling = ps.controls[idx_start+1:idx_end]
        sum(map(zip(F_i, local_sampling)) do (F_it, wit)
            F_it * wit
        end)
    end))

    return F
end

function fim(layer::OEDLayer{false}, p::AbstractArray)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sols, _ = layer(nothing, p + zero(ComponentArray(ps)), st)
    fim(layer.layers, sols)
end

function fim(layer::OEDLayer{false, <:SingleShootingLayer, <:Any, <:Any}, sols::DiffEqArray)
    fim(layer.layers, sols)
end

function fim(layer::OEDLayer{false, <:MultipleShootingLayer, <:Any, <:Any}, sols::EnsembleSolution)
    sum(map(enumerate(layer.layers)) do (i,_layer)
        fim(_layer, sols[i])
    end)
end