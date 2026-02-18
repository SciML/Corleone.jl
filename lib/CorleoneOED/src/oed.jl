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

default_observed = (u, p, t) -> u

"""
$(TYPEDEF)

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
    measurement_text = DISCRETE ? "discrete " : "continuous "
    print(
        io,
        no_color, layer_text,
        type_color, "OEDLayer ", no_color, "with ",
        type_color, measurement_text, #"with $(dims.nh) observation functions and $(dims.np_fisher) considered parameters.\n",
        no_color, "measurement model ",
        no_color, "and ", type_color, "$(size(sampling_indices, 1)) ", no_color, "observed functions.\n"
    )
    print(io, no_color, "Underlying problem: ")
    return Base.show(io, "text/plain", isa(layer, SingleShootingLayer) ? layer.problem : layer.layer.problem)
end

function OEDLayer{DISCRETE}(prob::DEProblem, alg::DEAlgorithm; params = eachindex(prob.p), measurements = [], observed = default_observed, kwargs...) where {DISCRETE}
    layer = SingleShootingLayer(prob, alg; kwargs...)
    return OEDLayer{DISCRETE}(layer; params = params, measurements = measurements, observed = observed, kwargs...)
end

function OEDLayer{DISCRETE}(prob::DEProblem, alg::DEAlgorithm, shooting_points...; params = eachindex(prob.p), measurements = [], observed = default_observed, kwargs...) where {DISCRETE}
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

n_observed(layer::OEDLayer) = length(layer.sampling_indices)
Corleone.get_number_of_shooting_constraints(oed::OEDLayer{<:Any, <:Any, <:Any, <:MultipleShootingLayer}) = Corleone.get_number_of_shooting_constraints(oed.layer)
Corleone.get_number_of_shooting_constraints(oed::OEDLayer{<:Any, <:Any, <:Any, <:SingleShootingLayer}) = 0
Corleone.get_bounds(oed::OEDLayer; kwargs...) = Corleone.get_bounds(oed.layer; kwargs...)

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

    return merge(
        st, (;
            observation_grid = WeightedObservation(weighting_grid),
            active_controls = measurement_indices,
        )
    )
end

function LuxCore.initialstates(rng::Random.AbstractRNG, oed::OEDLayer{true, true, <:Any, <:MultipleShootingLayer})
    (; layer, sampling_indices) = oed
    (; problem, controls, control_indices) = layer.layer
    st = LuxCore.initialstates(rng, layer)
    # Our goal is to build a weigthing matrix similar to the indexgrid
    grids = Corleone.get_timegrid.(controls)

    return map(st) do sti
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
fisher_information(oed::OEDLayer{true, true, false, <:SingleShootingLayer}, x, ps, st::NamedTuple) = begin
    (; observation_grid) = st
    traj, st = oed(x, ps, st)
    Gs = __fisher_information(oed, traj)
    observation_grid(ps.controls, Gs), st
end

# FIXED DISCRETE and SAMPLING -> use helper function
fisher_information(oed::OEDLayer{true, true, true}, x, ps, st::NamedTuple) = begin
    traj, st = oed(x, ps, st)
    __fisher_information(oed, traj, ps, st), st
end

fisher_information(oed::OEDLayer{true, true, false, <:MultipleShootingLayer}, x, ps, st::NamedTuple) = begin
    traj, st = oed(x, ps, st)
    Gs = __fisher_information(oed, traj, ps, st)
    sum(
            map(eachindex(Gs)) do i
                psi, sti = getproperty(ps, Symbol("interval_$i")), getproperty(st, Symbol("interval_$i"))
                sti.observation_grid(psi.controls, Gs[i])
        end
        ), st

end

# DISCRETE -> SUM
fisher_information(oed::OEDLayer{true, false}, x, ps, st::NamedTuple) = begin
    (; sampling_indices, layer) = oed
    (; observation_grid) = st
    traj, st = oed(x, ps, st)
    sum(__fisher_information(oed, traj)), st
end

# FIXED + CONTINUOUS
fisher_information(oed::OEDLayer{false, true, true}, x, ps, st::NamedTuple) = begin
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

get_sampling_sums(::OEDLayer{<:Any, false}, x, ps, st) = []
get_sampling_sums!(res, ::OEDLayer{<:Any, false}, x, ps, st) = nothing

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
