struct ObservedLayer{N,L,O} <: LuxCore.AbstractLuxWrapper{(:layer,)}
    "The name of the layer, used for display and logging purposes."
    name::N
    "The wrapped shooting layer used to produce trajectories."
    layer::L
    "The user-provided expression."
    observed::O
end

_extract_timepoints(x::Number) = [x]
_extract_timepoints(x::Expr) = begin 
    @assert x.head == :vect "Timepoints must be provided as a scalar or vector, e.g. `x(1.0)` or `x([1.0, 2.0])"
    reduce(vcat, x.args)
end

_collect_timepoints!(::Dict{Symbol, <:AbstractVector}, ::Symbol) = nothing

function _collect_timepoints!(collector::Dict{Symbol, <:AbstractVector}, ex::Expr)
    if ex.head == :call
        if ex.args[1] ∈ keys(collector)
            append!(collector[ex.args[1]], _extract_timepoints(ex.args[2]))
        else 
            for arg in ex.args
                _collect_timepoints!(collector, arg)
            end
        end
    end
    return 
end

_extract_timeindex(x::Number, indices) = indices[x]
_extract_timeindex(x::Expr, indices) = begin 
    @assert x.head == :vect "Timepoints must be provided as a scalar or vector, e.g. `x(1.0)` or `x([1.0, 2.0])"
    reduce(vcat, map(Base.Fix2(_extract_timeindex, indices), x.args))
end

replace_timepoints(x::Symbol, replacer) = x

function replace_timepoints(x::Expr, replacer::Dict{Symbol, <:Dict})
    if x.head == :call
        if x.args[1] ∈ keys(replacer)
            return Expr(:call, :getindex, x.args[1], 
                _extract_timeindex(x.args[2], replacer[x.args[1]])
            )
        else 
            return Expr(x.head, map(arg -> replace_timepoints(arg, replacer), x.args)...)
        end
    end
    return 
end

function find_indices(points, grid)
    t0, tinf = extrema(grid)
    Dict(map(points) do p 
        p <= t0 && return p => firstindex(grid)
        p >= tinf && return p => lastindex(grid)
        p => searchsortedlast(grid, p)
    end...)
end

function ObservedLayer(layer, expressions...; name=gensym(:observed))
    problem = Corleone.get_problem(layer) 
    symbols = vcat(variable_symbols(problem), parameter_symbols(problem))
    tspan = get_tspan(layer)
    collector = Dict([vi => eltype(tspan)[] for vi in symbols]) 
    foreach(expressions) do ex 
        _collect_timepoints!(collector, ex)
    end
    # Find the indices 
    timegrid = Corleone.get_timegrid(layer)
    foreach(values(collector)) do tps
        append!(timegrid, tps)
    end
    unique!(sort!(timegrid))
    layer = remake(layer, saveat = timegrid)
    replacer = Dict([ki => find_indices(vi, timegrid) for (ki, vi) in zip(keys(collector), values(collector))])
    new_exprs = map(expressions) do ex 
        replace_timepoints(ex, replacer)
    end
    returns = [gensym() for _ in expressions]
    exprs = [:($(returns[i]) = $(new_exprs[i])) for i in eachindex(returns)]
    # Minimal variable set 
    for (k, v) in zip(keys(collector), values(collector))
        if !isempty(v)
            if is_parameter(problem, k)
            pushfirst!(exprs, :($k = traj.ps[$(QuoteNode(k))]))
            else 
                pushfirst!(exprs, :($k = getsym(traj, $(QuoteNode(k)))(traj)))
            end
        end
    end
    push!(exprs, :(return [$(returns...),]))
    # Define the function header 
    expr = :((traj) -> ($(exprs...)))
    observed = @RuntimeGeneratedFunction(expr)

    return ObservedLayer{typeof(name),typeof(layer),typeof(observed_nt)}(name, layer, observed)
end

function (obs::ObservedLayer)(x, ps, st)
    trajectory, st = obs.layer(x, ps, st)
    observations = obs.observed(trajectory) 
    return (; observations, trajectory), st
end

