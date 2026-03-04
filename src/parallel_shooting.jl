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
struct MultipleShootingLayer{L, S <: NamedTuple} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    "The instance of a [ParallelShootingLayer](@ref) to be solved in parallel."
    layer::L 
    "Indicator for shooting constraints for each of the layers."
    shooting_variables::S
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
    layers = NamedTuple{ntuple(i->Symbol(:layer_,i), length(layers))}(layers) 
    i = 0 
    shooting_variables = map(layers) do layer 
        if i > 0  
            get_shooting_variables(layer) 
        else
            i += 1
            []
        end
    end
    layer = ParallelShootingLayer(layers; kwargs...)

    MultipleShootingLayer{typeof(layer), typeof(shooting_variables)}(layer, shooting_variables)
end

function SciMLBase.remake(layer::MultipleShootingLayer; kwargs...)
    layer = remake(layer.layer; kwargs...)
    MultipleShootingLayer{typeof(layer), typeof(layer.shooting_variables)}(layer, layer.shooting_variables)
end

function (layer::MultipleShootingLayer)(u0, ps, st)
    results, st = layer.layer(u0, ps, st)
    return Trajectory(layer, results), st
end

function _reduce_controls(control, controls...)
    prev_signals = _reduce_controls(controls...)
    map(zip(control.collection, prev_signals)) do (a, b)
        DiffEqArray(
            vcat(a.u, b.u), 
            vcat(a.t, b.t)
        )
    end
end

_reduce_controls(control) = control.collection

function reduce_controls(controls...)
    signals = _reduce_controls(controls...)
    return ParameterTimeseriesCollection(
        signals, 
        first(controls).paramcache
    )
end

function Trajectory(layer::MultipleShootingLayer, results::NamedTuple{fields}; 
    kwargs...
    ) where fields

    sys = symbolic_container(first(results))
    p = first(results).p

    quadratures = get_quadrature_indices(layer)
    quad_idx = Base.Fix1(variable_index, sys).(quadratures)
    q0 = zero(first(results)[quadratures][1])
    

    u = reduce(vcat, map(enumerate(results)) do (i, res)
        unew = map(res.u) do uj 
            l = 0 
            [k ∈ quad_idx ?  uj[k] + q0[l+=1] : uj[k] for k in eachindex(uj)]
        end
        q0 = unew[end][quad_idx]
        i == lastindex(results) ? unew : unew[1:end-1]
    end)

    t = reduce(vcat, map(enumerate(results)) do (i, res)
        i == lastindex(results) ? res.t : res.t[1:end-1]
    end)
    controls = reduce_controls(map(Base.Fix2(getfield, :controls), values(results))...)
    utype = eltype(first(results).u[1])
    shooting_violations = map(Base.OneTo(length(results)-1)) do i 
        u_next = results[i+1]
        u_prev = results[i]
        vars = layer.shooting_variables[i+1]
        (; 
            u0 = utype.(u_prev[vars.u0][end] .- u_next[vars.u0][1]),
            p = isempty(vars.p) ? utype[] : u_prev.ps[vars.p] .- u_next.ps[vars.p],
            controls = isempty(vars.controls) ? utype[] : u_prev.ps[vars.controls][end] .- u_next.ps[vars.controls][1],
        )
    end
    shooting_violations = (; 
        u0 = reduce(vcat, map(sv -> sv.u0, shooting_violations)),
        p = reduce(vcat, map(sv -> sv.p, shooting_violations)),
        controls = reduce(vcat, map(sv -> sv.controls, shooting_violations)),
    )
    Trajectory{typeof(sys), typeof(u), typeof(p), typeof(t), typeof(controls), typeof(shooting_violations)}(sys, u, p, t, controls, shooting_violations)
end

