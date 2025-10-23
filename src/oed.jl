"""
$(TYPEDEF)
Defines a callable layer for optimal experimental design purposes following a linearization-based
approach, augmenting the original system dynamics with the forward sensitivities of the
parameters of interest and the Fisher information matrix.
Boolean `fixed` describes whether states and sensitivities are constant, e.g., due to fixed
initial conditions and controls. In this case, the OED problem is much simpler.
Boolean `discrete` (default false) describes whether measurements are taken at discrete
time points.

# Fields
$(FIELDS)
"""
struct OEDLayer{fixed,discrete,L,O,D} <: LuxCore.AbstractLuxLayer
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
Per default, continuous measurements are taken for which the keyword `dt` specifies
the sampling grid discretization.
If `measurement_points` are supplied, discrete measurements can be taken at these points.
"""
function OEDLayer(prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.AbstractDEAlgorithm;
            measurement_points = nothing,
            control_indices = Int64[],
            controls = nothing,
            tunable_ic = Int64[],
            bounds_ic = nothing,
            observed = prob.f.observed == SciMLBase.DEFAULT_OBSERVED ? (u,p,t) -> u[eachindex(prob.u0)] : prob.f.observed,
            dt = (-)(reverse(prob.tspan)...)/100,
            params = setdiff(eachindex(prob.p), control_indices),
            kwargs...)

    layer = SingleShootingLayer(prob, alg, control_indices, controls;
                                tunable_ic = tunable_ic, bounds_ic=bounds_ic,
                                kwargs...)
    OEDLayer(layer; observed=observed, params=params, dt=dt, measurement_points=measurement_points)
end

"""
$(SIGNATURES)
Constructs a multiple shooting OEDLayer from an AbstractDEProblem, where the starts of
the shooting intervals are supplied via `shooting_points`.
Parameters of interest are supplied via indices of `prob.p` and the oberved function
is supplied via `observed` with signature (u,p,t).
Per default, continuous measurements are taken for which the keyword `dt` specifies
the sampling grid discretization.
If `measurement_points` are supplied, discrete measurements can be taken at these points.
"""
function OEDLayer(prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.AbstractDEAlgorithm,
            shooting_points;
            measurement_points=nothing,
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

    OEDLayer(layer; observed=observed, params=params, dt=dt, measurement_points=measurement_points)
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
function is_fixed(layer::OEDLayer{true})
    true
end

function is_fixed(layer::OEDLayer{false})
    false
end

"""
$(SIGNATURES)
General constructor for OEDLayer from a SingleShootingLayer or MultipleShootingLayer.
"""
function OEDLayer(layer::Union{SingleShootingLayer,MultipleShootingLayer};
                    observed = (u,p,t) -> u,
                    params = get_params(layer),
                    dt = (-)(reverse(tspan)...)/100,
                    measurement_points = nothing
                    )

    prob = get_problem(layer)
    _, control_indices = get_controls(layer)
    nx, np, nc, np_considered = length(prob.u0), length(prob.p), length(control_indices), length(params)

    fixed = is_fixed(layer)
    discrete = !isnothing(measurement_points)
    oed_layer = augment_layer_for_oed(layer, params=params, observed=observed, dt=dt, measurement_points=measurement_points)

    obs = begin
        x, p, t = Symbolics.variables(:x, 1:nx), Symbolics.variables(:p, 1:np), Symbolics.variable(:t)

        h = observed(x,p,t)
        hx = Symbolics.jacobian(h, x)
        hx_fun = Symbolics.build_function(hx, x, p, t, expression = Val{false}, cse=true)[1]

        (h = observed, hx = hx_fun)
    end

    dimensions = (np = np, nh = length(observed(prob.u0, prob.p, first(prob.tspan))),
                  np_fisher = np_considered, nc = nc, nx = nx)
    return OEDLayer{fixed, discrete, typeof(oed_layer), typeof(obs), typeof(dimensions)}(oed_layer, obs, dimensions)
end

LuxCore.initialparameters(rng::Random.AbstractRNG, layer::OEDLayer) = LuxCore.initialparameters(rng, layer.layer)
LuxCore.initialstates(rng::Random.AbstractRNG, layer::OEDLayer) = LuxCore.initialstates(rng, layer.layer)

function (layer::OEDLayer)(::Any, ps, st)
    layer.layer(nothing, ps, st)
end

function (init::AbstractNodeInitialization)(rng::AbstractRNG, layer::OEDLayer; kwargs...)
    init(rng, layer.layer; kwargs...)
end

"""
    get_bounds(layer)
Return lower and upper bounds of all variables associated to `layer`.
"""
get_bounds(layer::OEDLayer) = get_bounds(layer.layer)
get_shooting_constraints(layer::OEDLayer{false, <:Any, <:MultipleShootingLayer, <:Any, <:Any}) = get_shooting_constraints(layer.layer)
get_block_structure(layer::OEDLayer) = get_block_structure(layer.layer)
sensitivity_variables(layer::OEDLayer) = sensitivity_variables(layer.layer)
fisher_variables(layer::OEDLayer) = fisher_variables(layer.layer)
observed_sensitivity_product_variables(layer::OEDLayer, observed_idx::Int) = observed_sensitivity_product_variables(layer.layer, observed_idx)

### Functions to evaluate Fisher information matrices
function fim(oedlayer::OEDLayer{true, false})
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

    (p, ::Any) -> let Fs=Fs, ax=getaxes(ComponentArray(ps)), nc=nc
        ps = ComponentArray(p, ax)
        F = symmetric_from_vector(sum(map(zip(Fs, nc[1:end-1], nc[2:end])) do (F_i, idx_start, idx_end)
            local_sampling = ps.controls[idx_start+1:idx_end]
            sum(map(zip(F_i, local_sampling)) do (F_it, wit)
                F_it * wit
            end)
        end))
    end
end

function fim(oedlayer::OEDLayer{true, true})
    ps, st = LuxCore.setup(Random.default_rng(), oedlayer)
    sols, _ = oedlayer(nothing, ps, st)
    nc = vcat(0, cumsum(map(x -> length(x.t), oedlayer.layer.controls))...)
    Fs = map(enumerate(oedlayer.layer.controls)) do (i,sampling) # All fixed -> only sampling controls
        Gi = sensitivity_variables(oedlayer)
        idxs = findall(x -> x in sampling.t, sols.t)
        sol_t = sols[idxs]
        sol_Gs = sols[Gi][idxs]
        map(zip(sol_t, sol_Gs, sampling.t)) do (sol, sol_Gi, ti)
            gram = oedlayer.observed.hx(sol[1:oedlayer.dimensions.nx], oedlayer.layer.problem.p, ti)[i:i,:] * sol_Gi
            gram' * gram
        end
    end

    (p, ::Any) -> let Fs=Fs, ax=getaxes(ComponentArray(ps)), nc=nc
        ps = ComponentArray(p, ax)
        F = sum(map(zip(Fs, nc[1:end-1], nc[2:end])) do (F_i, idx_start, idx_end)
            local_sampling = ps.controls[idx_start+1:idx_end]
            sum(map(zip(F_i, local_sampling)) do (F_it, wit)
                F_it * wit
            end)
        end)
    end
end

function fim(oedlayer::OEDLayer{false, true})
    ps, st = LuxCore.setup(Random.default_rng(), oedlayer)
    ax = getaxes(ComponentArray(ps))
    nc = vcat(0, cumsum(map(x -> length(x.t), oedlayer.layer.controls))...)
    nh = oedlayer.dimensions.nh
    (p, ::Any) -> let ax=ax, oedlayer=oedlayer, nc=nc, nh=nh
        ps = ComponentArray(p, ax)
        sols, _ = oedlayer(nothing, ps, st)

        Fs = map(enumerate(oedlayer.layer.controls[end-nh+1:end])) do (i,sampling) # Last nh controls are sampling
            Gi = sensitivity_variables(oedlayer)
            idxs = findall(x -> x in sampling.t, sols.t)
            sol_t = sols[idxs]
            sol_Gs = sols[Gi][idxs]
            map(zip(sol_t, sol_Gs, sampling.t)) do (sol, sol_Gi, ti)
                gram = oedlayer.observed.hx(sol[1:oedlayer.dimensions.nx], oedlayer.layer.problem.p, ti)[i:i,:] * sol_Gi
                gram' * gram
            end
        end

        F = sum(map(zip(Fs, nc[end-nh:end-1], nc[end-nh+1:end])) do (F_i, idx_start, idx_end)
            local_sampling = ps.controls[idx_start+1:idx_end]
            sum(map(zip(F_i, local_sampling)) do (F_it, wit)
                F_it * wit
            end)
        end)

        return F
    end
end

function fim(oedlayer::Union{OEDLayer{true,false},OEDLayer{false,true},OEDLayer{true,true}}, p::AbstractArray)
    feval = fim(oedlayer)
    F = feval(p, nothing)
    return F
end

function fim(layer::SingleShootingLayer, sols::DiffEqArray)
    f_sym = Corleone.fisher_variables(layer)
    Corleone.symmetric_from_vector(sols[f_sym][end])
end

function fim(layer::MultipleShootingLayer, sols::EnsembleSolution)
    f_sym = Corleone.fisher_variables(layer)
    Corleone.symmetric_from_vector(last(sols)[f_sym][end])
end

function fim(layer::OEDLayer{false, false})
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    (p, ::Any) -> let ax=getaxes(ComponentArray(ps)), oedlayer=layer
        ps = ComponentArray(p, ax)
        sols, _ = oedlayer(nothing, ps, st)
        fim(oedlayer.layer, sols)
    end
end

function fim(layer::OEDLayer{false, false}, p::AbstractArray)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sols, _ = layer(nothing, p + zero(ComponentArray(ps)), st)
    fim(layer.layer, sols)
end

"""
$(METHODLIST)
Compute Fisher information matrix for given solution of layer `sols`.
"""
function fim(layer::OEDLayer{false, <:Any, <:SingleShootingLayer, <:Any, <:Any}, sols::DiffEqArray)
    fim(layer.layer, sols)
end

function fim(layer::OEDLayer{false, <:Any, <:MultipleShootingLayer, <:Any, <:Any}, sols::EnsembleSolution)
    fim(last(layer.layer.layers), last(sols))
end

### Methods to evaluate AbstractCriterion on different layers
(crit::AbstractCriterion)(oedlayer::OEDLayer{true, false, <:SingleShootingLayer, <:Any, <:Any}) = begin
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

(crit::AbstractCriterion)(oedlayer::OEDLayer{true, true, <:SingleShootingLayer, <:Any, <:Any}) = begin
    ps, st = LuxCore.setup(Random.default_rng(), oedlayer)
    sols, _ = oedlayer(nothing, ps, st)
    nc = vcat(0, cumsum(map(x -> length(x.t), oedlayer.layer.controls))...)
    Fs = map(enumerate(oedlayer.layer.controls)) do (i,sampling) # All fixed -> only sampling controls
        Gi = sensitivity_variables(oedlayer)
        idxs = findall(x -> x in sampling.t, sols.t)
        sol_t = sols[idxs]
        sol_Gs = sols[Gi][idxs]
        map(zip(sol_t, sol_Gs, sampling.t)) do (sol, sol_Gi, ti)
            gram = oedlayer.observed.hx(sol[1:oedlayer.dimensions.nx], oedlayer.layer.problem.p, ti)[i:i,:] * sol_Gi
            gram' * gram
        end
    end

    (p, ::Any) -> let Fs = Fs, ax = getaxes(ComponentArray(ps)), nc=nc
        ps = ComponentArray(p, ax)
        F = sum(map(zip(Fs, nc[1:end-1], nc[2:end])) do (F_i, idx_start, idx_end)
            local_sampling = ps.controls[idx_start+1:idx_end]
            sum(map(zip(F_i, local_sampling)) do (F_it, wit)
                F_it * wit
            end)
        end)
        crit(F)
    end
end

function (crit::AbstractCriterion)(oedlayer::OEDLayer{false, true})
    (p, ::Any) -> let layer=oedlayer
        crit(fim(layer, p))
    end
end

(crit::AbstractCriterion)(layer::SingleShootingLayer, sols::DiffEqArray) = begin
    crit(fim(layer, sols))
end

(crit::AbstractCriterion)(layer::MultipleShootingLayer, sols::EnsembleSolution) = begin
    crit(fim(layer, sols))
end

(crit::AbstractCriterion)(oedlayer::OEDLayer{false, false}) = begin
    (p, ::Any) -> let layer=oedlayer
        F = fim(layer, p)
        crit(F)
    end
end
