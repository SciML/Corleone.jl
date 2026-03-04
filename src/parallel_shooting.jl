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

get_quadrature_indices(layer::MultipleShootingLayer) = get_quadrature_indices(first(values(layer.layer.layers)))
get_problem(layer::MultipleShootingLayer) = get_problem(first(values(layer.layer.layers)))
get_tspan(layer::MultipleShootingLayer) = extrema(reduce(vcat, map(get_tspan, values(layer.layer.layers))))
get_tunable_u0(layer::MultipleShootingLayer) = get_tunable_u0(first(values(layer.layer.layers)))
get_tunable_p(layer::MultipleShootingLayer) = get_tunable_p(first(values(layer.layer.layers)))

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


"""
$(SIGNATURES)

Construct a single [`Trajectory`](@ref) from the per-segment trajectories returned by the
`ParallelShootingLayer` wrapped inside `layer`. The function:

- concatenates state arrays `u` and time arrays `t` across segments (omitting the
  duplicated endpoint of each intermediate segment),
- accumulates quadrature states across segment boundaries,
- records state and parameter shooting (continuity) violations at each interior node as a
  `NamedTuple` keyed by the segment field names, and
- merges the per-segment `ParameterTimeseriesCollection` control timeseries into one.
"""
function Trajectory(layer::MultipleShootingLayer, results::NamedTuple{fields}; 
    kwargs...
    ) where {fields}

    us = map(state_values, values(results))
    ts = map(current_time, values(results))

    p_getter = getsym(first(values(results)), get_tunable_p(layer))
    quads = get_quadrature_indices(layer)
    state_getter = getsym(first(values(results)), filter(∉(quads), variable_symbols(get_problem(layer))))
    control_getter = getsym(first(values(results)), get_control_symbols(first(values(layer.layer.layers))))
    p = p_getter(first(values(results)))

    quadrature_indices = Base.Fix1(variable_index, first(values(results))).(quads)
    q_prev = us[1][end][quadrature_indices]
    for i in eachindex(us)[2:end]
        for j in eachindex(us[i])
            us[i][j][quadrature_indices] += q_prev
        end
        q_prev = us[i][end][quadrature_indices]
    end

    unew = reduce(
        vcat, map(i -> i == lastindex(us) ? us[i] : us[i][1:(end - 1)], eachindex(us))
    )
    tnew = reduce(
        vcat, map(i -> i == lastindex(ts) ? ts[i] : ts[i][1:(end - 1)], eachindex(ts))
    )
    shooting_val_1 = ((u0 = eltype(first(unew))[], p = eltype(p)[], controls = eltype(first(first(unew)))[]))
    shooting_vals = map(eachindex(Base.front(fields))) do i
        uprev = results[fields[i]]
        unext = results[fields[i + 1]]

        (
            u0 = last(state_getter(uprev)) .- first(state_getter(unext)),
            p = p_getter(uprev) .- p_getter(unext),
            controls = last(control_getter(uprev)) .- first(control_getter(unext)),
        )
    end
    shootings = NamedTuple{fields}(
        (shooting_val_1, shooting_vals...)
    )
    controls = map(get_control_symbols(first(values(layer.layer.layers)))) do k 
        cget = getsym(first(values(results)), k)
        DiffEqArray(
            reduce(vcat, map(cget, values(results)))[1:end-1],
            tnew 
        )
    end

    Trajectory(
        first(results).sys, 
        unew, p, 
        tnew, 
        ParameterTimeseriesCollection(controls, deepcopy(p)),
        shootings
    )
end


function (layer::MultipleShootingLayer)(u0, ps, st)
    results, st = layer.layer(u0, ps, st)
    return results#Trajectory(layer, results), st
end
