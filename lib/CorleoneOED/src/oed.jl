# Helper for weighting the controls over the trajectory
struct WeightedObservation
    grid::Vector{Vector{Int64}}
end

function (w::WeightedObservation)(controls::AbstractVector{T}, i::Int64, G::AbstractArray) where {T}
    psub = [iszero(i) ? zero(T) : controls[i] for i in w.grid[i]]
    G = psub .* G
    G'G
end

function (w::WeightedObservation)(controls::AbstractVector{T}, G::AbstractVector{<:AbstractArray}) where {T}
    sum(eachindex(G)) do i
        w(controls, i, G[i])
    end
end

"""
$(TYPEDEF)

# Fields
$(FIELDS)
"""
struct OEDLayer{DISCRETE,SAMPLED,FIXED,L,O} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    "The underlying layer"
    layer::L
    "The observed functions"
    observed::O
    "The sampling indices"
    sampling_indices::Vector{Int64}
end

function Base.show(io::IO, oed::OEDLayer{DISCRETE,SAMPLED,FIXED}) where {DISCRETE,SAMPLED,FIXED}
    (; layer, observed, sampling_indices) = oed
    type_color, no_color = SciMLBase.get_colorizers(io)
    layer_text = FIXED ? "Fixed " : ""
    measurement_text = DISCRETE ? "discrete " : "continuous "
    print(io,
        no_color, layer_text,
        type_color, "OEDLayer ", no_color, "with ",
        type_color, measurement_text, #"with $(dims.nh) observation functions and $(dims.np_fisher) considered parameters.\n",
        no_color, "measurement model ",
        no_color, "and ", type_color, "$(size(sampling_indices, 1)) ", no_color, "observed functions.\n")
    print(io, no_color, "Underlying problem: ")
    Base.show(io, "text/plain", layer.problem)
end

# TODO: WRITE A CONSTRUCTOR FOR AN OEDLAYER FROM A MULTIPLESHOOTINGLAYER
function OEDLayer{DISCRETE}(layer::MultipleShootingLayer, args...; measurements=[], kwargs...) where {DISCRETE}
    OEDLayer{DISCRETE}(layer.layer, args...; measurements=measurements, kwargs...)
end

function OEDLayer{DISCRETE}(layer::L, args...; measurements=[], kwargs...) where {DISCRETE,L}

    (; problem, algorithm, controls, control_indices, tunable_ic, bounds_ic, state_initialization, bounds_p, parameter_initialization) = layer

    FIXED = isempty(control_indices) && isempty(tunable_ic)
    SAMPLED = !isempty(measurements)
    mode = DISCRETE ? (SAMPLED ? Val{:DiscreteSampled}() : Val{:Discrete}()) : (SAMPLED ? Val{:ContinuousSampled}() : Val{:Continuous}())
    p_length = length(problem.p)
    samplings = SAMPLED ? collect(eachindex(measurements)) : Int64[]
    ctrls = vcat(collect(control_indices .=> controls), samplings .+ p_length .=> measurements)
    samplings = samplings .+ length(controls)

    newproblem, observed = augment_system(mode, problem, algorithm;
        tunable_ic=copy(tunable_ic),
        control_indices=copy(control_indices), fixed=FIXED,
        kwargs...)

    # Replace the saveat with the sampling times
    saveats = if SAMPLED
        ts = reduce(vcat, Corleone.get_timegrid.(measurements))
        unique!(sort!(ts))
    else
        collect(problem.tspan)
    end
    newproblem = remake(newproblem, saveat=saveats)

    lb, ub = copy.(bounds_ic)
    for i in eachindex(newproblem.u0)
        i <= lastindex(problem.u0) && continue
        push!(lb, zero(eltype(newproblem.u0)))
        push!(ub, zero(eltype(newproblem.u0)))
    end
    newlayer = SingleShootingLayer(
        newproblem, algorithm; controls=ctrls, tunable_ic=copy(tunable_ic), bounds_ic=(lb, ub), state_initialization, bounds_p, parameter_initialization
    )

    OEDLayer{DISCRETE,SAMPLED,FIXED,typeof(newlayer),typeof(observed)}(newlayer, observed, samplings)
end

Corleone.get_bounds(oed::OEDLayer; kwargs...) = Corleone.get_bounds(oed.layer; kwargs...)

# This is the only case where we need to sample the trajectory
function LuxCore.initialstates(rng::Random.AbstractRNG, oed::Union{OEDLayer{true,true}, OEDLayer{false,true,true}})
    (; layer, sampling_indices) = oed
    (; problem, controls, control_indices) = layer
    st = LuxCore.initialstates(rng, layer)
    # Our goal is to build a weigthing matrix similar to the indexgrid
    grids = Corleone.get_timegrid.(controls)
    overall_grid = vcat(reduce(vcat, grids), collect(problem.tspan))
    unique!(sort!(overall_grid))
    observed_grid = map(grids[sampling_indices]) do grid
        unique!(sort!(grid))
        findall(∈(grid), overall_grid)
    end
    measurement_indices = Corleone.build_index_grid(controls...; problem.tspan, subdivide=100)
    measurement_indices = map(eachrow(measurement_indices[sampling_indices, :])) do mi
        unique(mi)
    end
    # Lets order this by time
    weighting_grid = map(eachindex(overall_grid)) do i
        map(eachindex(observed_grid)) do j
            id = findfirst(i .== observed_grid[j])
            isnothing(id) && return 0
            measurement_indices[j][id]
        end
    end
    @info weighting_grid
    merge(st, (; observation_grid=WeightedObservation(weighting_grid), active_controls=measurement_indices))
end

__fisher_information(oed::OEDLayer, traj::Trajectory) = oed.observed.fisher(traj)

function __fisher_information(oed::OEDLayer{<:Any, true, true}, traj::Trajectory, ps, st::NamedTuple)
    (; controls) = ps
    (; active_controls) = st
    fim = __fisher_information(oed, traj)

    diffF = eachslice.(diff(fim), dims=3)
    sum(map(enumerate(active_controls)) do (i, subset)
        wi = controls[subset]
        Fi = [F[i] for F in diffF]
        sum(Fi .* wi)
    end)
end

fisher_information(oed::OEDLayer, x, ps, st::NamedTuple) = begin
    traj, st = oed(x, ps, st)
    sum(__fisher_information(oed, traj)), st
end

# Continuous ALWAYS last FIM
fisher_information(oed::OEDLayer{false}, x, ps, st::NamedTuple) = begin
    traj, st = oed(x, ps, st)
    last(__fisher_information(oed, traj)), st
end

# DISCRETE and SAMPLING -> weighted sum
fisher_information(oed::OEDLayer{true,true}, x, ps, st::NamedTuple) = begin
    (; sampling_indices, layer) = oed
    (; observation_grid) = st
    traj, st = oed(x, ps, st)
    Gs = __fisher_information(oed, traj)
    observation_grid(ps.controls, Gs), st
end

# DISCRETE -> SUM
fisher_information(oed::OEDLayer{true,false}, x, ps, st::NamedTuple) = begin
    (; sampling_indices, layer) = oed
    (; observation_grid) = st
    traj, st = oed(x, ps, st)
    sum(__fisher_information(oed, traj)), st
end

# FIXED
fisher_information(oed::OEDLayer{false,true,true}, x, ps, st::NamedTuple) = begin
    (; sampling_indices, layer) = oed
    (; observation_grid) = st
    traj, st = oed(x, ps, st)
    __fisher_information(oed, traj, ps, st), st
end

sensitivities(oed::OEDLayer, traj::Trajectory) = oed.observed.sensitivities(traj)

sensitivities(oed::OEDLayer, x, ps, st::NamedTuple) = begin
    traj, st = oed(x, ps, st)
    sensitivities(oed, traj), st
end

observed_equations(oed::OEDLayer, traj::Trajectory) = oed.observed.observed(traj)

observed_equations(oed::OEDLayer, x, ps, st::NamedTuple) = begin
    traj, st = oed(x, ps, st)
    observed_equations(oed, traj), st
end

_local_information_gain(oed::OEDLayer, traj::Trajectory) = oed.observed.local_weighted_sensitivity(traj)

local_information_gain(oed::OEDLayer, x, ps, st::NamedTuple) = begin
    traj, st = oed(x, ps, st)
    # This returns hx G but stacked as a matrix [h_1_x G; h_2_x G; ...]
    hxGs = _local_information_gain(oed, traj)
    map(hxGs) do hxGi
        map(axes(hxGi, 1)) do i
            xi = hxGi[i:i, :]
            xi'xi
        end
    end, st
end

global_information_gain(oed::OEDLayer, x, ps, st::NamedTuple) = begin
    traj, st = oed(x, ps, st)
    F_tf, st = fisher_information(oed, x, ps, st)
    C = inv(F_tf)
    # This returns hx G but stacked as a matrix [h_1_x G; h_2_x G; ...]
    hxGs = _local_information_gain(oed, traj)
    map(hxGs) do hxGi
        map(axes(hxGi, 1)) do i
            xi = hxGi[i:i, :] * C
            xi'xi
        end
    end, st
end

get_sampling_sums(::OEDLayer{<:Any,false}, x, ps, st) = []
get_sampling_sums!(res, ::OEDLayer{<:Any,false}, x, ps, st) = nothing

__get_subsets(active_controls::AbstractVector, indices) = active_controls[indices]
__get_subsets(index_grid::AbstractMatrix, indices) = index_grid[indices, :]
__get_subsets(active_controls::Tuple, indices) = reduce(vcat, map(Base.Fix2(__get_subsets, indices), active_controls))
__get_subsets(active_controls::Tuple{AbstractMatrix,Vararg{AbstractMatrix}}, indices) = reduce(hcat, map(Base.Fix2(__get_subsets, indices), active_controls))

__get_dts(tspans::Tuple{Vararg{Tuple{<:Real,<:Real}}}) = vcat(first.(Base.front(tspans))..., collect(last(tspans))...)
__get_dts(tspans::Tuple) = reduce(vcat, map(eachindex(tspans)) do i
    i == lastindex(tspans) ? __get_dts(tspans[i]) : __get_dts(Base.front(tspans[i]))
end)

_get_dts(tspans) = diff(__get_dts(tspans))

get_sampling_sums(oed::OEDLayer, x, ps, st) = _get_sampling_sums(oed, x, ps, st)
get_sampling_sums!(res, oed::OEDLayer, x, ps, st) = _get_sampling_sums!(res, oed, x, ps, st, Val{true}())

function get_sampling_sums(oed::OEDLayer{<:Any,<:Any,<:Any,Corleone.MultipleShootingLayer}, x, ps, st::NamedTuple{fields}) where {fields}
    sum(fields) do f
        _get_sampling_sums(oed, x, getproperty(ps, f), getproperty(st, f))
    end
end

function get_sampling_sums!(res, oed::OEDLayer{<:Any,<:Any,<:Any,Corleone.MultipleShootingLayer}, x, ps, st::NamedTuple{fields}) where {fields}
	foreach(enumerate(fields)) do (i,f)
		_get_sampling_sums!(res, oed, x, getproperty(ps, f), getproperty(st, f), Val{i==1}())
    end
end

function _get_sampling_sums(oed::OEDLayer{true,true}, x, ps, st)
    (; sampling_indices) = oed
    (; active_controls) = st
    (; controls) = ps
    map(__get_subsets(active_controls, sampling_indices)) do subset
        sum(controls[subset])
    end
end

function _get_sampling_sums(oed::OEDLayer{false,true,true}, x, ps, st)
    (; active_controls, tspans) = st
    (; controls) = ps
    dts = _get_dts(tspans)
    map(active_controls) do subset
        sum(controls[subset] .* dts)
    end
end

function _get_sampling_sums!(res, oed::OEDLayer{false,true,true}, x, ps, st, ::Val{RESET}) where {RESET}
    (; active_controls, tspans) = st
    (; controls) = ps
    dts = _get_dts(tspans)
    foreach(enumerate(active_controls)) do (i,subset)
        res[i] = sum(controls[subset] .* dts)
    end
end

function _get_sampling_sums!(res::AbstractArray, oed::OEDLayer{true,true}, x, ps, st, ::Val{RESET}) where {RESET}
    (; sampling_indices) = oed
    (; active_controls) = st
    (; controls) = ps
    foreach(enumerate(__get_subsets(active_controls, sampling_indices))) do (i, subset)
        if RESET
            res[i] = sum(controls[subset])
        else
            res[i] += sum(controls[subset])
        end
    end
end

function _get_sampling_sums(oed::OEDLayer{false,true,false}, x, ps, st)
    (; sampling_indices,) = oed
    (; index_grid, tspans) = st
    (; controls) = ps
    dts = _get_dts(tspans)
    map(enumerate(eachrow(__get_subsets(index_grid, sampling_indices)))) do (i, subset)
        sum(controls[subset] .* dts)
    end
end

function _get_sampling_sums!(res::AbstractVector, oed::OEDLayer{false,true,false}, x, ps, st, ::Val{RESET}) where {RESET}
    (; sampling_indices,) = oed
    (; index_grid, tspans) = st
    (; controls) = ps
    dts = _get_dts(tspans)
    foreach(enumerate(eachrow(__get_subsets(index_grid, sampling_indices)))) do (i, subset)
        if RESET
            res[i] = sum(controls[subset] .* dts)
        else
            res[i] += sum(controls[subset] .* dts)
        end
    end
end

#==
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

function is_discrete(layer::OEDLayer{<:Any, true})
    true
end

function is_discrete(layer::OEDLayer{<:Any, false})
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
function fim(oedlayer::OEDLayer{true, false, <:SingleShootingLayer})
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
        symmetric_from_vector(sum(map(zip(Fs, nc[1:end-1], nc[2:end])) do (F_i, idx_start, idx_end)
            local_sampling = ps.controls[idx_start+1:idx_end]
            sum(map(zip(F_i, local_sampling)) do (F_it, wit)
                F_it * wit
            end)
        end))
    end
end

function fim(oedlayer::OEDLayer{true, true, <:SingleShootingLayer})
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
        sum(map(zip(Fs, nc[1:end-1], nc[2:end])) do (F_i, idx_start, idx_end)
            local_sampling = ps.controls[idx_start+1:idx_end]
            sum(map(zip(F_i, local_sampling)) do (F_it, wit)
                F_it * wit
            end)
        end)
    end
end

function fim(oedlayer::OEDLayer{false, true, <:SingleShootingLayer})
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

        sum(map(zip(Fs, nc[end-nh:end-1], nc[end-nh+1:end])) do (F_i, idx_start, idx_end)
            local_sampling = ps.controls[idx_start+1:idx_end]
            sum(map(zip(F_i, local_sampling)) do (F_it, wit)
                F_it * wit
            end)
        end)
    end
end


function fim(oedlayer::Union{OEDLayer{true,false,<:SingleShootingLayer},OEDLayer{false,true,<:SingleShootingLayer},OEDLayer{true,true,<:SingleShootingLayer}}, p::AbstractArray)
    feval = fim(oedlayer)
    feval(p, nothing)
end

function fim(oedlayer::OEDLayer{false, true, <:MultipleShootingLayer})
    ps, st = LuxCore.setup(Random.default_rng(), oedlayer)
    ax = getaxes(ComponentArray(ps))
    nc = [vcat(0, cumsum(map(x -> length(x.t), layer.controls))...) for layer in oedlayer.layer.layers]
    nh = oedlayer.dimensions.nh
    (p, ::Any) -> let ax=ax, oedlayer=oedlayer, nc=nc, nh=nh
        ps = ComponentArray(p, ax)
        sols, _ = oedlayer(nothing, ps, st)

        Fs = map(1:nh) do i # Last nh controls are sampling
            map(enumerate(oedlayer.layer.layers)) do (idx_layer, layer)
                sampling = layer.controls[end-nh+i]
                Gi = Corleone.sensitivity_variables(oedlayer)
                idxs = findall(x -> x in sampling.t, sols[idx_layer].t)
                sol_t = sols[idx_layer][idxs]
                sol_Gs = sols[idx_layer][Gi][idxs]
                map(zip(sol_t, sol_Gs, sampling.t)) do (sol, sol_Gi, ti)
                    gram = oedlayer.observed.hx(sol[1:oedlayer.dimensions.nx], Corleone.get_problem(oedlayer.layer).p, ti)[i:i,:] * sol_Gi
                    gram' * gram
                end
            end
        end

        sum(map(1:nh) do i
            sum(map(1:length(oedlayer.layer.shooting_intervals)) do idx_layer
                local_sampling = getproperty(ps, Symbol("layer_$idx_layer"))
                idx_start, idx_end = nc[idx_layer][end-nh-1+i:end-nh+i]
                w_i_j = local_sampling.controls[idx_start+1:idx_end]
                sum(Fs[i][idx_layer] .* w_i_j)
            end)
        end)
    end
end

function fim(oedlayer::OEDLayer{false,true,<:MultipleShootingLayer}, p::AbstractArray)
    feval = fim(oedlayer)
    feval(p, nothing)
end

function fim(layer::SingleShootingLayer, sols::DiffEqArray)
    f_sym = Corleone.fisher_variables(layer)
    Corleone.symmetric_from_vector(sols[f_sym][end])
end

function fim(layer::MultipleShootingLayer, sols::EnsembleSolution)
    f_sym = Corleone.fisher_variables(layer)
    Corleone.symmetric_from_vector(last(sols)[f_sym][end])
end

function fim(layer::Union{OEDLayer{false, false},OEDLayer{<:Any,false, <:MultipleShootingLayer}})
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    (p, ::Any) -> let ax=getaxes(ComponentArray(ps)), oedlayer=layer
        ps = ComponentArray(p, ax)
        sols, _ = oedlayer(nothing, ps, st)
        fim(oedlayer.layer, sols)
    end
end

function fim(layer::Union{OEDLayer{false, false},OEDLayer{<:Any,<:Any, <:MultipleShootingLayer}}, p::AbstractArray)
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    sols, _ = layer(nothing, p + zero(ComponentArray(ps)), st)
    fim(layer.layer, sols)
end

"""
$(METHODLIST)
Compute Fisher information matrix for given solution of layer `sols`.
"""
function fim(layer::OEDLayer{false, false, <:SingleShootingLayer, <:Any, <:Any}, sols::DiffEqArray)
    fim(layer.layer, sols)
end

function fim(layer::OEDLayer{false, false, <:MultipleShootingLayer, <:Any, <:Any}, sols::EnsembleSolution)
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


## Functions to evaluate maximum sampling constraints
function control_blocks(controls::ControlParameter...)
    return vcat(0, cumsum([length(x.t) for x in controls]))
end

function control_blocks(layer::SingleShootingLayer)
    control_blocks(get_controls(layer)[1]...)
end

function control_blocks(layer::OEDLayer{<:Any, <:Any, <:SingleShootingLayer})
    control_blocks(layer.layer)
end

function control_blocks(layer::OEDLayer{<:Any, <:Any, <:MultipleShootingLayer})
    map(layer.layer.layers) do _layer
        control_blocks(_layer)
    end
end

function get_sampling_constraint(layer::OEDLayer{<:Any, false, <:SingleShootingLayer})
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    p = ComponentArray(ps)
    controls, _ = get_controls(layer.layer)

    nc = control_blocks(layer)
    diffs_t = [diff(x.t) for x in controls]
    dt = [all(y -> ≈(y,difft[1]), difft) ? difft[1] : difft  for difft in diffs_t]

    sampling_cons = let ax = getaxes(p), nc = nc, dt = dt, dims=layer.dimensions
        (p, ::Any) -> begin
            ps = ComponentArray(p, ax)
            [sum(ps.controls[nc[i]+1:nc[i+1]]) .* dt[i] for i in eachindex(nc)[dims.nc+1:end-1]]
        end
    end

    return sampling_cons
end

function get_sampling_constraint(layer::OEDLayer{<:Any, false, <:MultipleShootingLayer})
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    p = ComponentArray(ps)

    controls = first.(get_controls.(layer.layer.layers))
    nc = control_blocks(layer)

    diffs_layer = [[diff(x.t) for x in control] for control in controls]
    dt = [[all(y -> ≈(y,difft[1]), difft) ? difft[1:1] : difft  for difft in diffs_t] for diffs_t in diffs_layer]

    sampling_cons = let ax = getaxes(p), nc = nc, dt = dt, dims=layer.dimensions
        (p, ::Any) -> begin
            ps = ComponentArray(p, ax)
            reduce(vcat, map(1:dims.nh) do i
                sum([sum(ps["layer_$j"].controls[nc[j][dims.nc+i]+1:nc[j][dims.nc+i+1]]) .* _dt[i] for (j,_dt) in enumerate(dt)])
            end)
        end
    end

    return sampling_cons
end


function get_sampling_constraint(layer::OEDLayer{<:Any, true, <:SingleShootingLayer})
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    p = ComponentArray(ps)

    nc = control_blocks(layer)
    sampling_cons = let ax = getaxes(p), nc = nc, dims=layer.dimensions
        (p, ::Any) -> begin
            ps = ComponentArray(p, ax)
            [sum(ps.controls[nc[i]+1:nc[i+1]]) for i in eachindex(nc)[dims.nc+1:end-1]]
        end
    end

    return sampling_cons
end

function get_sampling_constraint(layer::OEDLayer{<:Any, true, <:MultipleShootingLayer})
    ps, st = LuxCore.setup(Random.default_rng(), layer)
    p = ComponentArray(ps)

    nc = control_blocks(layer)
    sampling_cons = let ax = getaxes(p), nc = nc, dims=layer.dimensions
        (p, ::Any) -> begin
            ps = ComponentArray(p, ax)
            reduce(vcat, map(1:dims.nh) do i
                sum([sum(ps["layer_$j"].controls[_nc[dims.nc+i]+1:_nc[dims.nc+i+1]])  for (j,_nc) in enumerate(nc)])
            end)
        end
    end

    return sampling_cons
end
==#
