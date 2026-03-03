struct ParallelShootingLayer{L <: NamedTuple, A <: SciMLBase.EnsembleAlgorithm} <: LuxCore.AbstractLuxWrapperLayer{:layers}
    name::Symbol
    "The layers to be solved in parallel. Each layer should be a SingleShootingLayer."
    layers::L
    "The underlying ensemble algorithm to use for parallelization. Default is `EnsembleThreads`."
    ensemble_algorithm::A
end

ParallelShootingLayer(layers::NamedTuple; kwargs...) = ParallelShootingLayer(
    get(kwargs, :name, gensym(:parallel_shooting)), 
    layers, 
    get(kwargs, :ensemble_algorithm, EnsembleSerial()))

function ParallelShootingLayer(layers::AbstractLuxLayer...; kwargs...)
    @assert all(is_shooting_layer, layers) "All layers must be shooting layers."
    layers = NamedTuple{ntuple(i->Symbol(:layer,i), length(layers))}(layers)
    ParallelShootingLayer(layers; kwargs...)
end

get_quadrature_indices(layer::ParallelShootingLayer) = get_quadrature_indices(first(layer.layers))

function get_block_structure(layer::ParallelShootingLayer)
    return vcat(0, cumsum(map(LuxCore.parameterlength, layer.layers)))
end

function (layer::ParallelShootingLayer)(u0, ps, st)
    _parallel_solve(layer.ensemble_algorithm, layer.layers, u0, ps, st)
end

__getidx(x, id) = x[id]
__getidx(x::NamedTuple, id) = getproperty(x, id)

function _parallel_solve(
        alg::SciMLBase.EnsembleAlgorithm,
        layers::NamedTuple{fields},
        u0,
        ps,
        st::NamedTuple{fields},
    ) where {fields}

    args = ntuple(
            i -> (__getidx(layers, fields[i]), u0, __getidx(ps, fields[i]), __getidx(st, fields[i])), length(st)
        )
    
    ret =  mythreadmap(alg, Base.splat(LuxCore.apply), args)
    return NamedTuple{fields}(first.(ret)), NamedTuple{fields}(last.(ret))
end

function SciMLBase.remake(layer::ParallelShootingLayer; kwargs...)
    layers = map(keys(layer.layers)) do k 
        layer_kwargs = get(kwargs, k, kwargs)
        k, remake(layer.layers[k]; layer_kwargs...)
    end |> NamedTuple
    ensemble_algorithm = get(kwargs, :ensemble_algorithm, layer.ensemble_algorithm)
    ParallelShootingLayer(layer.name, layers, ensemble_algorithm)
end

"""
$(TYPEDEF)

Defines a layer for multiple shooting. Simply a wrapper for the [ParallelShootingLayer](@ref) but returns a single trajectory.
"""
struct MultipleShootingLayer{L} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    "The instance of a [ParallelShootingLayer](@ref) to be solved in parallel."
    layer::L 
end

function MultipleShootingLayer(layer::LuxCore.AbstractLuxLayer, shooting_points::Real...; kwargs...)
    @assert is_shooting_layer(layer) "The provided layer must be a shooting layer."
    problem = get_problem(layer)
    tspan = get_tspan(layer)
    quadratures = get_quadrature_indices(layer)
    tunables = setdiff(variable_symbols(problem), quadratures)
    tpoints = unique!(sort!(vcat(collect(shooting_points), collect(tspan))))
    layers = ntuple(i -> remake(layer, 
        tspan = (tpoints[i], tpoints[i+1]),
        tunable_u0 = i == 1 ? get_tunable_u0(layer) : tunables,
    ), length(tpoints)-1)
    layers = NamedTuple{ntuple(i->Symbol(:layer,i), length(layers))}(layers) 
    layer = ParallelShootingLayer(layers; kwargs...)
    MultipleShootingLayer{typeof(layer)}(layer)
end

function SciMLBase.remake(layer::MultipleShootingLayer; kwargs...)
    layer = remake(layer.layer; kwargs...)
    MultipleShootingLayer{typeof(layer)}(layer)
end

get_quadrature_indices(layer::MultipleShootingLayer) = get_quadrature_indices(layer.layer)

function (layer::MultipleShootingLayer)(u0, ps, st)
    results, new_st = layer.layer(u0, ps, st)
    quadrature_indices = get_quadrature_indices(layer)
    traj = _combine_shooting_trajectories(results, quadrature_indices)
    return traj, new_st
end

"""
    _combine_shooting_trajectories(results::NamedTuple, quadrature_indices)

Combine the per-interval trajectories from a `MultipleShootingLayer` into a single
`Trajectory` with shooting constraint violations stored in the `shooting` field.
"""
function _combine_shooting_trajectories(results::NamedTuple{fields}, quadrature_indices) where {fields}
    trajs = values(results)
    length(trajs) == 1 && return only(trajs)

    p = first(trajs).p
    sys = first(trajs).sys
    us = [collect(state_values(traj)) for traj in trajs]
    ts = [collect(current_time(traj)) for traj in trajs]

    # Combined time: remove the duplicate endpoint between consecutive intervals
    tnew = reduce(
        vcat, [i == lastindex(ts) ? ts[i] : ts[i][1:(end - 1)] for i in eachindex(ts)]
    )

    # Compute shooting constraint violations (state and parameter matching at boundaries).
    # The first interval has empty violations (it is the reference interval).
    T_elem = eltype(first(first(us)))
    non_quad = isempty(quadrature_indices) ? eachindex(first(first(us))) :
               setdiff(eachindex(first(first(us))), quadrature_indices)
    shooting_val_1 = (; u0=T_elem[], p=eltype(p)[], controls=T_elem[])
    shooting_vals = map(eachindex(us)[1:(end - 1)]) do i
        uprev = us[i]
        unext = us[i + 1]
        (;
            u0=last(uprev)[non_quad] .- first(unext)[non_quad],
            p=trajs[i].p .- trajs[i + 1].p,
            controls=T_elem[],
        )
    end
    shootings = NamedTuple{fields}((shooting_val_1, shooting_vals...))

    # Accumulate quadrature across intervals (running sum of quadrature states)
    if !isempty(quadrature_indices)
        q_prev = us[1][end][quadrature_indices]
        for i in eachindex(us)[2:end]
            for j in eachindex(us[i])
                us[i][j][quadrature_indices] .+= q_prev
            end
            q_prev = us[i][end][quadrature_indices]
        end
    end

    unew = reduce(
        vcat, [i == lastindex(us) ? us[i] : us[i][1:(end - 1)] for i in eachindex(us)]
    )

    # Combine per-interval control timeseries into one for the full trajectory
    n_controls = length(first(trajs).controls.collection)
    combined_series = ntuple(n_controls) do k
        all_u = reduce(vcat, [traj.controls.collection[k].u for traj in trajs])
        all_t = reduce(vcat, [traj.controls.collection[k].t for traj in trajs])
        DiffEqArray(all_u, all_t)
    end
    controls_combined = ParameterTimeseriesCollection(combined_series, deepcopy(p))

    return Trajectory(sys, unew, p, tnew, controls_combined, shootings)
end
