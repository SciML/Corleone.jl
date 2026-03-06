struct ParallelShootingLayer{L<:NamedTuple,A<:SciMLBase.EnsembleAlgorithm} <: LuxCore.AbstractLuxWrapperLayer{:layers}
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
    layers = NamedTuple{ntuple(i -> Symbol(:layer, i), length(layers))}(layers)
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

    ret = mythreadmap(alg, Base.splat(LuxCore.apply), args)
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
struct MultipleShootingLayer{L,S<:NamedTuple} <: LuxCore.AbstractLuxWrapperLayer{:layer}
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
            tspan=(tpoints[i], tpoints[i+1]),
            tunable_u0=i == 1 ? get_tunable_u0(layer) : tunables,
        ), length(tpoints) - 1)
    layers = NamedTuple{ntuple(i -> Symbol(:layer_, i), length(layers))}(layers)
    shooting_variables = map(get_shooting_variables, layers) 
    layer = ParallelShootingLayer(layers; kwargs...)

    MultipleShootingLayer{typeof(layer),typeof(shooting_variables)}(layer, shooting_variables)
end

function SciMLBase.remake(layer::MultipleShootingLayer; kwargs...)
    layer = remake(layer.layer; kwargs...)
    MultipleShootingLayer{typeof(layer),typeof(layer.shooting_variables)}(layer, layer.shooting_variables)
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

_subselect(u::AbstractArray{T}, idxs) where T = [i ∈ idxs ? u[i] : zero(T) for i in eachindex(u)]
_vecadd(u::AbstractArray{T}, v) where T = [u[i] .+ v for i in eachindex(u)]


@generated function __collect_multiple_shooting_solutions(sols::NamedTuple{fields}, quad_idxs) where {fields}
    us = [gensym(:u) for _ in fields]
    ts = [gensym(:t) for _ in fields]
    qs = [gensym(:q) for _ in fields]
    exprs = Expr[]
    for (i, f) in enumerate(fields)
        if i < lastindex(fields)
            if i > firstindex(fields)
                push!(exprs, :($(us[i]) = _vecadd(sols.$(f).u[1:end-1], $(qs[i]))))
            else
                push!(exprs, :($(us[i]) = sols.$(f).u[1:end-1]))
                push!(exprs, :($(qs[i]) = zero(sols.$(f).u[1])))
            end
            push!(exprs, :($(ts[i]) = sols.$(f).t[1:end-1]))
            push!(exprs, :($(qs[i+1]) = _subselect(sols.$(f).u[end], quad_idxs) + $(qs[i])))
        else
            push!(exprs, :($(us[i]) = _vecadd(sols.$(f).u, $(qs[i]))))
            push!(exprs, :($(ts[i]) = sols.$(f).t))
        end
    end
    push!(exprs, :(return (vcat($(us...)), vcat($(ts...)))))
    ex = Expr(:block, exprs...)
    return ex
end

@generated function shooting_constraints(variables::NamedTuple{fields}, prev, next) where {fields}
    results = [gensym() for i in 1:3]
    exprs = Expr[]
    for (i, k) in enumerate((:u0, :p, :controls))
        if k ∈ fields
            if k == :u0
                push!(exprs, :($(results[i]) = prev[variables.$(k)][end] .- next[variables.$(k)][1]))
            elseif k == :p && !isempty(variables.p)
                push!(exprs, :($(results[i]) = prev.ps[variables.$(k)] .- next.ps[variables.$(k)]))
            else # controls
                 push!(exprs, :($(results[i]) = prev.ps[variables.$(k)][end] .- next.ps[variables.$(k)][1]))
            end
        else
             push!(exprs, :($(results[i]) = utype(prev)[]))
        end
    end
    push!(exprs, :(($(results[1]), $(results[2]), $(results[3]))))
    Expr(:block, exprs...)
end

@generated function shooting_constraints(results::NamedTuple{fields}, shooting_vars::NamedTuple{fields}) where {fields}
    N = length(fields) - 1
    u0s = [gensym(:u0) for _ in 1:N]
    ps = [gensym(:p) for _ in 1:N]
    controls = [gensym(:controls) for _ in 1:N]
    exprs = Expr[]  
    for i in 1:N 
        push!(exprs, :(($(u0s[i]), $(ps[i]), $(controls[i])) = shooting_constraints(shooting_vars.$(fields[i+1]), 
            results.$(fields[i]), 
            results.$(fields[i+1])))
        )
    end
    push!(exprs, :(vcat($(u0s...)), vcat($(ps...)), vcat($(controls...))))
    Expr(:block, exprs...)
end

function Trajectory(layer::MultipleShootingLayer, results::NamedTuple{fields};
    kwargs...
) where fields

    sys = symbolic_container(first(results))
    p = first(results).p

    quadratures = get_quadrature_indices(layer)
    quad_idx = Base.Fix1(variable_index, sys).(quadratures)
    u, t = __collect_multiple_shooting_solutions(results, quad_idx)
    controls = reduce_controls(map(Base.Fix2(getfield, :controls), values(results))...)
    shooting_violations = shooting_constraints(results, layer.shooting_variables)
    Trajectory{typeof(sys),typeof(u),typeof(p),typeof(t),typeof(controls),typeof(shooting_violations)}(sys, u, p, t, controls, shooting_violations)
end

