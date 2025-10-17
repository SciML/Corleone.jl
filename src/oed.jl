"""
$(TYPEDEF)
Defines a callable layer for optimal experimental design purposes following a linearization-based
approach, augmenting the original system dynamics with the forward sensitivities of the
parameters of interest and the Fisher information matrix.
Boolean `fixed` describes whether states and sensitivities are constant, e.g., due to fixed
initial conditions and controls. In this case, the OED problem is much simpler.

# Fields
$(FIELDS)
"""
struct OEDLayer{fixed,L,O,D} <: LuxCore.AbstractLuxLayer
    "Either a SingleShootingLayer or a MultipleShootingLayer."
    layer::L
    "Callable observed function h(u,p,t) and its jacobian hx(u,p,t)."
    observed::O
    "Statistics on the number of states, number of parameters, etc."
    dimensions::D
end

"""
$(SIGNATURES)
Constructs a single shooting OEDLayer from an AbstractDEProblem.
Parameters of interest are supplied via indices of `prob.p` and the oberved function
is supplied via `observed` with signature (u,p,t).
Keyword `dt` specifies the sampling grid discretization.
"""
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

"""
$(SIGNATURES)
Constructs a multiple shooting OEDLayer from an AbstractDEProblem, where the starts of
the shooting intervals are supplied via `shooting_points`.
Parameters of interest are supplied via indices of `prob.p` and the oberved function
is supplied via `observed` with signature (u,p,t).
Keyword `dt` specifies the sampling grid discretization.
"""
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

"""
    is_fixed(layer)

Returns whether states of dynamical system of layer are constant due to fixed initial conditions
in the absence of controls.
"""
function is_fixed(layer::Union{SingleShootingLayer, MultipleShootingLayer})
    controls, control_indices = get_controls(layer)
    isempty(get_tunable(layer)) && (isempty(control_indices) || isnothing(controls))
end

"""
    is_fixed(layer)

Returns whether states and sensitivities OEDLayer are constant due to fixed initial conditions
and an absence of controls.
"""
function is_fixed(layer::OEDLayer{true, <:Any, <:Any, <:Any})
    true
end

function is_fixed(layer::OEDLayer{false, <:Any, <:Any, <:Any})
    false
end

"""
$(SIGNATURES)
General constructor for OEDLayer from a SingleShootingLayer or MultipleShootingLayer.
"""
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
"""
    get_bounds(layer)
Return lower and upper bounds of all variables associated to `layer`.
"""
get_bounds(layer::OEDLayer) = get_bounds(layer.layer)
get_shooting_constraints(layer::OEDLayer{false, <:MultipleShootingLayer, <:Any, <:Any}) = get_shooting_constraints(layer.layer)
get_block_structure(layer::OEDLayer) = get_block_structure(layer.layer)
sensitivity_variables(layer::OEDLayer) = sensitivity_variables(layer.layer)
fisher_variables(layer::OEDLayer) = fisher_variables(layer.layer)
observed_sensitivity_product_variables(layer::OEDLayer, observed_idx::Int) = observed_sensitivity_product_variables(layer.layer, observed_idx)

"""
$(METHODLIST)
Compute Fisher information matrix for given iterate `p`.
"""
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

function fim(oedlayer::OEDLayer{false}, p::AbstractArray)
    ps, st = LuxCore.setup(Random.default_rng(), oedlayer)
    sols, _ = oedlayer(nothing, p + zero(ComponentArray(ps)), st)
    fim(oedlayer.layer, sols)
end
"""
$(METHODLIST)
Compute Fisher information matrix for given solution of layer `sols`.
"""
function fim(oedlayer::OEDLayer{false, <:SingleShootingLayer, <:Any, <:Any}, sols::DiffEqArray)
    fim(oedlayer.layer, sols)
end

function fim(oedlayer::OEDLayer{false, <:MultipleShootingLayer, <:Any, <:Any}, sols::EnsembleSolution)
    sum(map(enumerate(oedlayer.layers)) do (i,_layer)
        fim(_layer, sols[i])
    end)
end