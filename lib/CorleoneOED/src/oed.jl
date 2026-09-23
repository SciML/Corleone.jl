#=
# Helper for weighting the controls over the trajectory
struct WeightedObservation
    grid::Vector{Vector{Int64}}
end

function (w::WeightedObservation)(controls::AbstractVector{T}, i::Int64, G::AbstractArray) where {T}
    psub = [iszero(j) ? zero(T) : controls[j] for j in w.grid[i]]
    G = psub .* G
    return G'G
end

function (w::WeightedObservation)(controls::AbstractVector{T}, G::AbstractVector{<:AbstractArray}) where {T}
    return sum(eachindex(G)) do i
        w(controls, i, G[i])
    end
end
=#

abstract type AbstractMeasurement end

@concrete struct DiscreteMeasurement <: AbstractMeasurement
    "Measurement identifier"
    id 
    "Discrete measurement points"
    tpoints
    "Observed function"
    observed
end

@concrete struct ContinuousMeasurement <: AbstractMeasurement 
    "Measurement identifier"
    id
    "Breakpoints of continuous measurement grid"
    tpoints
    "Observed function"
    observed
end

@concrete terse struct OEDLayer <: LuxCore.AbstractLuxContainerLayer{(:shooting,)}
    "The underlying shooting layer"
    shooting
    "Problem with augmented differential equations"
    augmented_prob
    "Measurements"
    measurements
end

function control_from_measurement(m::Union{DiscreteMeasurement,ContinuousMeasurement})
    return Corleone.PiecewiseParameter(
        Symbol(m.id), m.tpoints, 1.0, (ps, st) -> ([zeros(1,) for _ in 1:length(m.tpoints)+1], [ones(1,) for _ in 1: length(m.tpoints)+1])
    )
end

function get_missing_params(prob, control_symbols, params::Array{<:Symbol})
    return filter(p -> p ∉ control_symbols, params)
end

function get_missing_params(prob, control_symbols, params::Array{<:Int})
    map(params) do idx
        findfirst(==(idx), prob.f.sys.parameters)
    end
end

function OEDLayer(
    problem::SciMLBase.AbstractDEProblem,
    variable_id,
    params::Union{Vector{<:Int}, Vector{<:Symbol}},
    controls...;
    shooting_method::Corleone.AbstractAutoShoot = NoShoot(),
    algorithm::SciMLBase.AbstractDEAlgorithm,
    ensemble_algorithm::SciMLBase.EnsembleAlgorithm = EnsembleSerial(),
    tspan = problem.tspan,
    measurements = AbstractMeasurement[],
    kwargs...
)

    current_control_symbols = [cp.parameter_id for cp in controls if cp isa PiecewiseParameter]
    missing_params = begin
        try
            get_missing_params(problem, current_control_symbols, collect(keys(problem.f.sys.parameters)))
        catch
            get_missing_params(problem, current_control_symbols, eachindex(problem.p))
        end
    end 
    
    auto_controls = map(missing_params) do p_sym
       # Find the index of the parameter in the problem to get initial value
       p_idx = get(problem.f.sys.parameters, p_sym, nothing)
       val = p_idx !== nothing ? problem.p[p_idx] : 1.0
       PiecewiseParameter(p_sym, [problem.tspan[1]], val, (val, val))
    end

    observed_continuous = filter(x -> typeof(x) <: ContinuousMeasurement, measurements)
    observed_discrete = filter(x -> typeof(x) <: DiscreteMeasurement, measurements)

    newproblem, observed = augment_system(
        problem, algorithm, params = params, continuous_measurements = observed_continuous,
        discrete_measurements = observed_discrete
    )

    sampling_controls = PiecewiseParameter[CorleoneOED.control_from_measurement(x) for x in measurements]

    shooting_layer = ShootingLayer(newproblem, variable_id, controls..., auto_controls..., sampling_controls...,
        algorithm=algorithm, ensemble_algorithm=ensemble_algorithm, shooting_method=shooting_method,
        tspan = tspan
    )

    return OEDLayer(shooting_layer, newproblem, (; observed = observed, discrete = observed_discrete, continuous = observed_continuous))
end

function get_size_F(oed::OEDLayer)
    size_hxG = size(oed.measurements.observed.fisher.getters)
    if length(size_hxG) > 2
        size_hxG = size_hxG[1:2]
    end
    return (size_hxG[2], size_hxG[2])
end

LuxCore.initialparameters(rng::Random.AbstractRNG, oed::OEDLayer) = LuxCore.initialparameters(rng, oed.shooting)
LuxCore.initialstates(rng::Random.AbstractRNG, oed::OEDLayer) = begin
    
    st = LuxCore.initialstates(rng, oed.shooting)
    size_F = get_size_F(oed)
    T = eltype(oed.augmented_prob.u0)
    F_init = zeros(T, size_F)

    return merge(st, (; F_init = F_init))
end

function update_fim(oed::OEDLayer, experiments, st::NamedTuple)
    FIM = sum(
        map(experiments) do experiment
            # sum up only information gained by the experiment
            fisher_information(oed, nothing, experiment.ps, st)[1] - st.F_init
        end
    )

    return merge(st, (; F_init = FIM + st.F_init))
end

function update_fim(oed::OEDLayer, experiments)
    ps, st = LuxCore.setup(Random.default_rng(), oed)
    FIM = sum(
        map(experiments) do experiment
            # sum up only information gained by the experiment
            fisher_information(oed, nothing, experiment.ps, st)[1]
        end
    )

    return merge(st, (; F_init = FIM))
end

function (x::OEDLayer)(Nothing, ps, st)
    x.shooting(x.augmented_prob, ps, st)
end

function __continuous_fisher_information(oed::OEDLayer, traj::Trajectory)
    (; measurements) = oed
    (; observed, continuous) = measurements
    isempty(continuous) && return zeros(eltype(oed.augmented_prob.u0), get_size_F(oed))
    F_cont = last.(oed.measurements.observed.fisher(traj))
    return F_cont
end

function __discrete_fisher_information(oed::OEDLayer, traj::Trajectory)
    (; measurements) = oed
    (; observed, discrete) = measurements

    isempty(discrete) && return zeros(eltype(oed.augmented_prob.u0), get_size_F(oed))
    hx_G_discrete = observed.hx_G_discrete(traj)
    F_discrete = sum(map(enumerate(discrete)) do (i,obs_disc_i)
        id, tpoints = obs_disc_i.id, obs_disc_i.tpoints
        id_t_in_sol = [findfirst(t -> isapprox(t, ti; atol=1e-10, rtol=0), traj.t) for ti in tpoints]
        sol_w = traj[id][id_t_in_sol]
        sol_hx_G = [getindex.(hx_G_discrete, idx)[i:i,:] for idx in id_t_in_sol]
        F_disc = [x' * x for x in sol_hx_G]
        sum(sol_w .* F_disc)
    end)
    return F_discrete
end

__fisher_information(oed::OEDLayer, traj::Trajectory) = __discrete_fisher_information(oed, traj) .+ __continuous_fisher_information(oed, traj)

__fisher_information(oed::OEDLayer, x, ps, st::NamedTuple) = begin
    sol, _ = oed(x, ps, st)
    return __fisher_information(oed, sol)
end

function fisher_information(oed, x, ps, st::NamedTuple)
    sol, _ = oed(x, ps, st)
    st.F_init + __fisher_information(oed, sol), st
end


function discrete_sampling_sums(oed::OEDLayer, x, ps, st::NamedTuple)
    (; measurements,) = oed
    (; discrete,) = measurements

    ctrl = ps.controls.controls

    res = zeros(eltype(first(first(ps.controls.controls))), size(discrete, 1))
    @inbounds for i in eachindex(discrete)
        vecs = getproperty(ctrl, discrete[i].id)
        s = zero(eltype(first(first(ctrl))))
        for v in vecs
            start = (v === first(vecs)) ? 2 : 1
            @inbounds for j=start:length(v)
                s += v[j]
            end
        end
        res[i] = s

    end
    return res
end

function discrete_sampling_sums!(res, oed::OEDLayer, x, ps, st::NamedTuple)
    (; measurements,) = oed
    (; discrete,) = measurements
    ctrl = ps.controls.controls

    @inbounds for i in eachindex(discrete)
        vecs = getproperty(ctrl, discrete[i].id)
        s = zero(eltype(first(first(ctrl))))
        for v in vecs
            start = (v === first(vecs)) ? 2 : 1
            @inbounds for j=start:length(v)
                s += v[j]
            end
        end
        res[i] = s
    end
    return
end

function continuous_sampling_sums(oed::OEDLayer, x, ps, st::NamedTuple)
    (; measurements,) = oed
    (; continuous,) = measurements

    sol, _ = oed(x, ps, st)
    res = zeros(eltype(first(first(ps.controls.controls))), size(continuous, 1))
    dt = diff(sol.t)
    @inbounds for i in eachindex(continuous)
        w_i = sol[continuous[i].id]
        res[i] = dot(dt, @view w_i[1:end-1])
    end
    return res
end

function continuous_sampling_sums!(res, oed::OEDLayer, x, ps, st::NamedTuple)
    (; measurements,) = oed
    (; continuous,) = measurements

    sol, _ = oed(x, ps, st)
    dt = diff(sol.t)

    @inbounds for i in eachindex(continuous)
        w_i = sol[continuous[i].id]
        res[i] = dot(dt, @view w_i[1:end-1])
    end
    return
end

function sampling_sums(oed::OEDLayer, x, ps, st)
    return vcat(continuous_sampling_sums(oed, x, ps, st), discrete_sampling_sums(oed, x, ps, st))
end

function sampling_sums!(res, oed::OEDLayer, x, ps, st)
    n_cont, n_disc = length(oed.measurements.continuous), length(oed.measurements.discrete)
    if n_cont > 0
        continuous_sampling_sums!(view(res, (1:n_cont)), oed, x, ps, st)
    end
    if n_disc > 0
        discrete_sampling_sums!(view(res, (n_cont+1:n_cont+n_disc)), oed, x, ps, st)
    end
    return
end

Corleone.get_number_of_shooting_constraints(oed::OEDLayer) = Corleone.get_number_of_shooting_constraints(oed.shooting)
n_observed(layer::OEDLayer) = length(layer.measurements.discrete) + length(layer.measurements.continuous)

#=
"""
$(TYPEDEF)

Wraps a Corleone shooting layer with the augmented dynamics needed for optimal experimental
design. The `DISCRETE` type parameter selects discrete or continuous information accumulation,
and optional measurement controls can restrict where observations are collected.

# Fields
$(FIELDS)
"""
struct OEDLayer{DISCRETE, SAMPLED, FIXED, L, O} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    "The underlying layer"
    layer::L
    "The observed functions"
    observed::O
    "The sampling indices"
    sampling_indices::Vector{Int64}
end

is_fixed(oed::OEDLayer{<:Any, <:Any, T}) where {T} = T

function Base.show(io::IO, oed::OEDLayer{DISCRETE, SAMPLED, FIXED}) where {DISCRETE, SAMPLED, FIXED}
    (; layer, observed, sampling_indices) = oed
    type_color, no_color = SciMLBase.get_colorizers(io)
    layer_text = FIXED ? "Fixed " : ""
    preposition = SAMPLED ? "with " : "without "
    measurement_color = SAMPLED ? type_color : no_color
    measurement_text = SAMPLED ? (DISCRETE ? "discrete " : "continuous ") : "specified "
    print(
        io,
        no_color, layer_text,
        type_color, "OEDLayer ",
        no_color, preposition,
        measurement_color, measurement_text,
        no_color, "measurement model ",
        no_color, "and ", type_color, "$(size(sampling_indices, 1)) ", no_color, "observed functions.\n"
    )
    print(io, no_color, "Underlying problem: ")
    return Base.show(io, "text/plain", isa(layer, SingleShootingLayer) ? layer.problem : layer.layer.problem)
end



"""
$(SIGNATURES)

Constructs an `OEDLayer` from a differential equation problem and solver algorithm.

# Arguments
- `prob`: Differential equation problem whose parameters are considered for experimental design.
- `alg`: Differential equation solver algorithm.

# Keywords
- `params`: Parameter indices included in the Fisher information matrix.
- `measurements`: Optional `ControlParameter`s that define measurement schedules.
- `observed`: Observation map `(u, p, t) -> y`.

# Returns
An `OEDLayer` that can be initialized with `LuxCore.setup` and evaluated like the wrapped
Corleone layer.
"""
function OEDLayer{DISCRETE}(prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.AbstractDEAlgorithm; params = eachindex(prob.p), measurements = [], observed = default_observed, kwargs...) where {DISCRETE}
    layer = SingleShootingLayer(prob, alg; kwargs...)
    return OEDLayer{DISCRETE}(layer; params = params, measurements = measurements, observed = observed, kwargs...)
end

function OEDLayer{DISCRETE}(prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.AbstractDEAlgorithm, shooting_points...; params = eachindex(prob.p), measurements = [], observed = default_observed, kwargs...) where {DISCRETE}
    layer = MultipleShootingLayer(prob, alg, shooting_points...; kwargs...)
    return OEDLayer{DISCRETE}(layer; params = params, measurements = measurements, observed = observed, kwargs...)
end

function OEDLayer{DISCRETE}(layer::MultipleShootingLayer, args...; measurements = [], kwargs...) where {DISCRETE}

    (; problem, algorithm, controls, control_indices, tunable_ic, bounds_ic, state_initialization, bounds_p, parameter_initialization, quadrature_indices) = layer.layer

    FIXED = isempty(control_indices) && isempty(tunable_ic)
    SAMPLED = !isempty(measurements)
    mode = DISCRETE ? (SAMPLED ? Val{:DiscreteSampled}() : Val{:Discrete}()) : (SAMPLED ? Val{:ContinuousSampled}() : Val{:Continuous}())
    p_length = length(problem.p)
    samplings = SAMPLED ? collect(eachindex(measurements)) : Int64[]
    ctrls = vcat(collect(control_indices .=> controls), samplings .+ p_length .=> measurements)
    samplings = samplings .+ length(controls)

    newproblem, observed = augment_system(
        mode, problem, algorithm;
        tunable_ic = copy(tunable_ic),
        control_indices = copy(control_indices), fixed = FIXED,
        kwargs...
    )

    # Replace the saveat with the sampling times
    saveats = if SAMPLED
        ts = reduce(vcat, Corleone.get_timegrid.(measurements))
        unique!(sort!(ts))
    else
        collect(problem.tspan)
    end
    newproblem = remake(newproblem, saveat = saveats)

    shooting_points = [t[1] for t in layer.shooting_intervals]
    newlayer = MultipleShootingLayer(
        newproblem, algorithm, shooting_points...; controls = ctrls, tunable_ic = copy(tunable_ic), state_initialization, bounds_p, parameter_initialization, quadrature_indices = Int64[]
    )

    return OEDLayer{DISCRETE, SAMPLED, FIXED, typeof(newlayer), typeof(observed)}(newlayer, observed, samplings)
end

function OEDLayer{DISCRETE}(layer::SingleShootingLayer, args...; measurements = [], kwargs...) where {DISCRETE}

    (; problem, algorithm, controls, control_indices, tunable_ic, bounds_ic, state_initialization, bounds_p, parameter_initialization, quadrature_indices) = layer

    FIXED = isempty(control_indices) && isempty(tunable_ic)
    SAMPLED = !isempty(measurements)
    mode = DISCRETE ? (SAMPLED ? Val{:DiscreteSampled}() : Val{:Discrete}()) : (SAMPLED ? Val{:ContinuousSampled}() : Val{:Continuous}())
    p_length = length(problem.p)
    samplings = SAMPLED ? collect(eachindex(measurements)) : Int64[]
    ctrls = vcat(collect(control_indices .=> controls), samplings .+ p_length .=> measurements)
    samplings = samplings .+ length(controls)

    newproblem, observed = augment_system(
        mode, problem, algorithm;
        tunable_ic = copy(tunable_ic),
        control_indices = copy(control_indices), fixed = FIXED,
        kwargs...
    )

    # Replace the saveat with the sampling times
    saveats = if SAMPLED
        ts = reduce(vcat, Corleone.get_timegrid.(measurements))
        unique!(sort!(ts))
    else
        collect(problem.tspan)
    end
    newproblem = remake(newproblem, saveat = saveats)

    lb, ub = copy.(bounds_ic)
    for i in eachindex(newproblem.u0)
        i <= lastindex(problem.u0) && continue
        push!(lb, zero(eltype(newproblem.u0)))
        push!(ub, zero(eltype(newproblem.u0)))
    end
    newlayer = SingleShootingLayer(
        newproblem, algorithm; controls = ctrls, tunable_ic = copy(tunable_ic), bounds_ic = (lb, ub), state_initialization, bounds_p, parameter_initialization, quadrature_indices = Int64[]
    )

    return OEDLayer{DISCRETE, SAMPLED, FIXED, typeof(newlayer), typeof(observed)}(newlayer, observed, samplings)
end

function update_fim(oed::OEDLayer{<:Any, SAMPLED, FIXED, <:SingleShootingLayer}, experiments, st::NamedTuple) where {SAMPLED, FIXED}
    FIM = sum(
        map(experiments) do experiment
            fisher_information(oed, nothing, experiment.ps, experiment.st)[1]
        end
    )

    return merge(st, (; F_init = FIM + st.F_init))
end

function update_fim(oed::OEDLayer{<:Any, SAMPLED, FIXED, <:MultipleShootingLayer}, experiments, st::NamedTuple) where {SAMPLED, FIXED}
    FIM = sum(
        map(experiments) do experiment
            fisher_information(oed, nothing, experiment.ps, experiment.st)[1]
        end
    )

    st1 = merge(st[1], (; F_init = FIM + st[1].F_init))

    return merge(st, (; interval_1 = st1))
end

n_observed(layer::OEDLayer) = length(layer.sampling_indices)

Corleone.get_bounds(oed::OEDLayer; kwargs...) = Corleone.get_bounds(oed.layer; kwargs...)

get_size_F(oed::OEDLayer{true, true, <:Any}) = begin
    size_hxG = size(oed.observed.fisher.getters)
    if length(size_hxG) > 2
        size_hxG = size_hxG[1:2]
    end
    return (size_hxG[2], size_hxG[2])
end
get_size_F(oed::OEDLayer{false, true, true}) = begin
    sizeF = size(oed.observed.fisher.getters)
    if length(sizeF) > 2
        sizeF = sizeF[1:2]
    end
    return sizeF
end

# This is the only case where we need to sample the trajectory
function LuxCore.initialstates(rng::Random.AbstractRNG, oed::Union{OEDLayer{true, true, <:Any, <:SingleShootingLayer}, OEDLayer{false, true, true}})
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
    _measurement_indices = Corleone.build_index_grid(controls...; problem.tspan)
    measurement_indices = map(eachrow(_measurement_indices[sampling_indices, :])) do mi
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

    # in active controls, also the indices of the original, non-sampling controls must be added
    measurement_indices = typeof(oed) <: OEDLayer{true, true} ? begin
            indices_all_controls = collect(1:length(control_indices))
            map(eachrow(_measurement_indices[indices_all_controls, :])) do mi
                unique(mi)
        end
        end : measurement_indices

    T = eltype(problem.u0)
    F_init = zeros(T, get_size_F(oed))

    return merge(
        st, (;
            observation_grid = WeightedObservation(weighting_grid),
            active_controls = measurement_indices,
            F_init = F_init,
        )
    )
end

function LuxCore.initialstates(rng::Random.AbstractRNG, oed::OEDLayer{true, true, <:Any, <:MultipleShootingLayer})
    (; layer, sampling_indices) = oed
    (; problem, controls, control_indices) = layer.layer
    st = LuxCore.initialstates(rng, layer)
    # Our goal is to build a weigthing matrix similar to the indexgrid
    grids = Corleone.get_timegrid.(controls)
    T = eltype(problem.u0)
    F_init = zeros(T, get_size_F(oed))

    st_new = map(st) do sti
        tspan = (first(first(sti.tspans)), last(last(sti.tspans)))
        overall_grid = vcat(reduce(vcat, grids), collect(tspan))
        unique!(sort!(overall_grid))
        overall_grid = overall_grid[overall_grid .>= first(tspan) .&& overall_grid .< last(tspan)]
        observed_grid = map(grids[sampling_indices]) do grid
            unique!(sort!(grid))
            findall(∈(grid), overall_grid)
        end
        _measurement_indices = Corleone.build_index_grid(controls...; tspan = tspan)
        measurement_indices = map(eachrow(_measurement_indices[sampling_indices, :])) do mi
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

        # in active controls, also the indices of the original, non-sampling controls must be added
        measurement_indices = begin
            indices_all_controls = collect(1:length(control_indices))
            map(eachrow(_measurement_indices[indices_all_controls, :])) do mi
                unique(mi)
            end
        end

        merge(
            sti, (;
                observation_grid = WeightedObservation(weighting_grid),
                active_controls = measurement_indices,
            )
        )
    end

    st1 = merge(st_new[1], (; F_init = F_init))
    return merge(st_new, (; interval_1 = st1))
end

get_problem(oed::OEDLayer{<:Any, <:Any, <:Any, <:SingleShootingLayer}) = oed.layer.problem
get_problem(oed::OEDLayer{<:Any, <:Any, <:Any, <:MultipleShootingLayer}) = oed.layer.layer.problem

function LuxCore.initialstates(rng::Random.AbstractRNG, oed::OEDLayer{<:Any, <:Any, <:Any, <:SingleShootingLayer})
    (; layer, sampling_indices) = oed
    st = LuxCore.initialstates(rng, layer)
    problem = get_problem(oed)
    T = eltype(problem.u0)
    F_init = zeros(T, size(oed.observed.fisher.getters))

    return merge(st, (; F_init = F_init))
end

function LuxCore.initialstates(rng::Random.AbstractRNG, oed::OEDLayer{<:Any, <:Any, <:Any, <:MultipleShootingLayer})
    (; layer, sampling_indices) = oed
    st = LuxCore.initialstates(rng, layer)
    problem = get_problem(oed)
    T = eltype(problem.u0)
    F_init = zeros(T, size(oed.observed.fisher.getters))

    st1 = st.interval_1
    st1 = merge(st1, (; F_init = F_init))
    return merge(st, (; interval_1 = st1))
end

__fisher_information(oed::OEDLayer, traj::Trajectory) = oed.observed.fisher(traj)

function __fisher_information(oed::OEDLayer{true, true, false, <:MultipleShootingLayer}, traj::Trajectory, ps, st::NamedTuple)
    nc = vcat(
        0, cumsum(
            map(1:length(st)) do i
                sti = getproperty(st, Symbol("interval_$i"))
                length(sti.observation_grid.grid)
            end
        )
    )

    Gs = oed.observed.fisher(traj)
    return [Gs[(nc[i] + 1):nc[i + 1]] for i in 1:(size(nc, 1) - 1)]
end

function __fisher_information(oed::OEDLayer{false, true, true}, traj::Trajectory, ps, st::NamedTuple)
    (; controls) = ps
    (; active_controls) = st
    fim = __fisher_information(oed, traj)

    w = eachrow(reduce(hcat, map(x -> controls[x], active_controls)))
    diffF = map(-, fim[2:end], fim[1:end])

    return sum([F[:, :, k] .* wi[k] for (wi, F) in zip(w, diffF) for k in axes(F, 3)])
end

function __fisher_information(oed::OEDLayer{true, true, true}, traj::Trajectory, ps, st::NamedTuple)
    (; controls) = ps
    (; observation_grid) = st
    Gs = __fisher_information(oed, traj)
    return observation_grid(controls, Gs)
end

"""
$(SIGNATURES)

Computes the Fisher information matrix for an `OEDLayer` at parameters `ps` and state `st`.

# Arguments
- `oed`: Optimal experimental design layer.
- `x`: External input passed through the Lux layer interface.
- `ps`: Layer parameters, typically from `LuxCore.initialparameters` or `LuxCore.setup`.
- `st`: Layer state, typically from `LuxCore.initialstates` or `LuxCore.setup`.

# Keywords
- `add_initial`: Include previously accumulated information stored in `st`.

# Returns
A tuple `(F, st)` containing the Fisher information matrix and updated layer state.
"""
fisher_information(oed::OEDLayer, x, ps, st::NamedTuple; add_initial = true) = begin
    traj, st = oed(x, ps, st)
    F = add_initial ? st.F_init + sum(__fisher_information(oed, traj)) : sum(__fisher_information(oed, traj))
    return F, st
end

# Continuous ALWAYS last FIM
fisher_information(oed::OEDLayer{false, <:Any, <:Any, <:SingleShootingLayer}, x, ps, st::NamedTuple; add_initial = true) = begin
    traj, st = oed(x, ps, st)
    F = add_initial ? st.F_init + last(__fisher_information(oed, traj)) : last(__fisher_information(oed, traj))
    return F, st
end

fisher_information(oed::OEDLayer{false, <:Any, <:Any, <:MultipleShootingLayer}, x, ps, st::NamedTuple; add_initial = true) = begin
    traj, st = oed(x, ps, st)
    F = add_initial ? st[1].F_init + last(__fisher_information(oed, traj)) : last(__fisher_information(oed, traj))
    return F, st
end

# DISCRETE and SAMPLING -> weighted sum
fisher_information(oed::OEDLayer{true, true, false, <:SingleShootingLayer}, x, ps, st::NamedTuple; add_initial = true) = begin
    (; observation_grid) = st
    traj, st = oed(x, ps, st)
    Gs = __fisher_information(oed, traj)
    F = add_initial ? st.F_init + observation_grid(ps.controls, Gs) : observation_grid(ps.controls, Gs)
    return F, st
end

# FIXED DISCRETE and SAMPLING -> use helper function
fisher_information(oed::OEDLayer{true, true, true}, x, ps, st::NamedTuple; add_initial = true) = begin
    traj, st = oed(x, ps, st)
    F = add_initial ? st.F_init + __fisher_information(oed, traj, ps, st) : __fisher_information(oed, traj, ps, st)
    return F, st
end

fisher_information(oed::OEDLayer{true, true, false, <:MultipleShootingLayer}, x, ps, st::NamedTuple; add_initial = true) = begin
    traj, st = oed(x, ps, st)
    Gs = __fisher_information(oed, traj, ps, st)
    F = sum(
        map(eachindex(Gs)) do i
            psi, sti = getproperty(ps, Symbol("interval_$i")), getproperty(st, Symbol("interval_$i"))
            sti.observation_grid(psi.controls, Gs[i])
        end
    )

    add_initial && return st[1].F_init + F, st
    return F, st
end

# DISCRETE -> SUM
fisher_information(oed::OEDLayer{true, false, <:Any, <:SingleShootingLayer}, x, ps, st::NamedTuple; add_initial = true) = begin
    (; sampling_indices, layer) = oed
    (; observation_grid) = st
    traj, st = oed(x, ps, st)
    F = add_initial ? st.F_init + sum(__fisher_information(oed, traj)) : sum(__fisher_information(oed, traj))
    return F, st
end

fisher_information(oed::OEDLayer{true, false, <:Any, <:MultipleShootingLayer}, x, ps, st::NamedTuple; add_initial = true) = begin
    (; sampling_indices, layer) = oed
    (; observation_grid) = st
    traj, st = oed(x, ps, st)
    F = add_initial ? st[1].F_init + sum(__fisher_information(oed, traj)) : sum(__fisher_information(oed, traj))
    return F, st
end

# FIXED + CONTINUOUS
fisher_information(oed::OEDLayer{false, true, true}, x, ps, st::NamedTuple; add_initial = true) = begin
    (; sampling_indices, layer) = oed
    (; observation_grid) = st
    traj, st = oed(x, ps, st)
    F = add_initial ? st.F_init + __fisher_information(oed, traj, ps, st) : __fisher_information(oed, traj, ps, st)
    return F, st
end

"""
$(SIGNATURES)

Returns the parameter sensitivities of the observed outputs for an OED trajectory.

# Returns
For trajectory inputs, the sensitivity arrays are returned directly. For layer inputs, the
return value is `(sensitivities, st)` with the updated layer state.
"""
sensitivities(oed::OEDLayer, traj::Trajectory) = oed.observed.sensitivities(traj)

sensitivities(oed::OEDLayer, x, ps, st::NamedTuple) = begin
    traj, st = oed(x, ps, st)
    sensitivities(oed, traj), st
end

"""
$(SIGNATURES)

Evaluates the observation equations associated with an `OEDLayer`.

# Returns
For trajectory inputs, observed values are returned directly. For layer inputs, the return
value is `(observed, st)` with the updated layer state.
"""
observed_equations(oed::OEDLayer, traj::Trajectory) = oed.observed.observed(traj)

observed_equations(oed::OEDLayer, x, ps, st::NamedTuple) = begin
    traj, st = oed(x, ps, st)
    observed_equations(oed, traj), st
end

_local_information_gain(oed::OEDLayer, traj::Trajectory) = oed.observed.local_weighted_sensitivity(traj)

"""
$(SIGNATURES)

Computes local information gain contributions from each observed quantity along the OED
trajectory.

# Returns
A tuple `(gains, st)` containing per-observation information matrices and the updated layer
state.
"""
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

"""
$(SIGNATURES)

Computes global information gain contributions scaled by the inverse final Fisher information
matrix.

# Returns
A tuple `(gains, st)` containing per-observation information matrices and the updated layer
state.
"""
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

get_sampling_sums(::OEDLayer{<:Any, false}, x, ps, st) = []
get_sampling_sums!(res, ::OEDLayer{<:Any, false}, x, ps, st) = nothing
get_sampling_sums(::OEDLayer{<:Any, false, <:Any, <:Corleone.MultipleShootingLayer}, x, ps, st::NamedTuple{fields}) where {fields} = []
get_sampling_sums!(res, ::OEDLayer{<:Any, false, <:Any, <:Corleone.MultipleShootingLayer}, x, ps, st::NamedTuple{fields}) where {fields} = nothing

__get_subsets(active_controls::AbstractVector, indices) = active_controls[indices]
__get_subsets(index_grid::AbstractMatrix, indices) = index_grid[indices, :]
__get_subsets(active_controls::Tuple, indices) = reduce(vcat, map(Base.Fix2(__get_subsets, indices), active_controls))
__get_subsets(active_controls::Tuple{AbstractMatrix, Vararg{AbstractMatrix}}, indices) = reduce(hcat, map(Base.Fix2(__get_subsets, indices), active_controls))

__get_dts(tspans::Tuple{Vararg{Tuple{<:Real, <:Real}}}) = vcat(first.(Base.front(tspans))..., collect(last(tspans))...)
__get_dts(tspans::Tuple) = reduce(
    vcat, map(eachindex(tspans)) do i
        i == lastindex(tspans) ? __get_dts(tspans[i]) : __get_dts(Base.front(tspans[i]))
    end
)

_get_dts(tspans) = diff(__get_dts(tspans))

get_sampling_sums(oed::OEDLayer, x, ps, st) = _get_sampling_sums(oed, x, ps, st)
get_sampling_sums!(res, oed::OEDLayer, x, ps, st) = _get_sampling_sums!(res, oed, x, ps, st, Val{true}())

function get_sampling_sums(oed::OEDLayer{<:Any, <:Any, <:Any, <:Corleone.MultipleShootingLayer}, x, ps, st::NamedTuple{fields}) where {fields}
    return sum(fields) do f
        _get_sampling_sums(oed, x, getproperty(ps, f), getproperty(st, f))
    end
end

function get_sampling_sums!(res, oed::OEDLayer{<:Any, <:Any, <:Any, <:Corleone.MultipleShootingLayer}, x, ps, st::NamedTuple{fields}) where {fields}
    return foreach(enumerate(fields)) do (i, f)
        _get_sampling_sums!(res, oed, x, getproperty(ps, f), getproperty(st, f), Val{i == 1}())
    end
end

function _get_sampling_sums(oed::OEDLayer{true, true}, x, ps, st)
    (; sampling_indices) = oed
    (; active_controls) = st
    (; controls) = ps
    return map(__get_subsets(active_controls, sampling_indices)) do subset
        sum(controls[subset])
    end
end

function _get_sampling_sums(oed::OEDLayer{false, true, true}, x, ps, st)
    (; active_controls, tspans) = st
    (; controls) = ps
    dts = _get_dts(tspans)
    return map(active_controls) do subset
        sum(controls[subset] .* dts)
    end
end

function _get_sampling_sums!(res, oed::OEDLayer{false, true, true}, x, ps, st, ::Val{RESET}) where {RESET}
    (; active_controls, tspans) = st
    (; controls) = ps
    dts = _get_dts(tspans)
    return foreach(enumerate(active_controls)) do (i, subset)
        res[i] = sum(controls[subset] .* dts)
    end
end

function _get_sampling_sums!(res::AbstractArray, oed::OEDLayer{true, true}, x, ps, st, ::Val{RESET}) where {RESET}
    (; sampling_indices) = oed
    (; active_controls) = st
    (; controls) = ps
    return foreach(enumerate(__get_subsets(active_controls, sampling_indices))) do (i, subset)
        if RESET
            res[i] = sum(controls[subset])
        else
            res[i] += sum(controls[subset])
        end
    end
end

function _get_sampling_sums(oed::OEDLayer{false, true, false}, x, ps, st)
    (; sampling_indices) = oed
    (; index_grid, tspans) = st
    (; controls) = ps
    dts = _get_dts(tspans)
    return map(enumerate(eachrow(__get_subsets(index_grid, sampling_indices)))) do (i, subset)
        sum(controls[subset] .* dts)
    end
end

function _get_sampling_sums!(res::AbstractVector, oed::OEDLayer{false, true, false}, x, ps, st, ::Val{RESET}) where {RESET}
    (; sampling_indices) = oed
    (; index_grid, tspans) = st
    (; controls) = ps
    dts = _get_dts(tspans)
    return foreach(enumerate(eachrow(__get_subsets(index_grid, sampling_indices)))) do (i, subset)
        if RESET
            res[i] = sum(controls[subset] .* dts)
        else
            res[i] += sum(controls[subset] .* dts)
        end
    end
end

get_block_structure(layer::OEDLayer) = get_block_structure(layer.layer)
=#