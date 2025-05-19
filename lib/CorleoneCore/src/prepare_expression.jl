function prepare_timedependent_expressions(sys, exprs...)
    exprs = Symbolics.unwrap.(exprs)
    subs = [] 
    exceptions = unknowns(sys)
    exprs = map(exprs) do ex 
        replace_timedeps!(ex, subs; exceptions) 
    end
    timepoints = collect_timepoints(subs) 
    subs, orig = find_subexpressions(subs)
    (;
        expressions = collect(exprs), 
        timepoints = timepoints, 
        timereducers = subs, 
        reducermap = orig
    )
end

replace_timedeps!(ex::Number, subs; kwargs...) = ex 
replace_timedeps!(ex::AbstractVector{<: Number}, subs; kwargs...) = ex

function replace_timedeps!(ex::SymbolicUtils.Symbolic, subs; exceptions=[])
    if CorleoneCore.is_timepointterm(ex)
        newvar = Symbolics.variable(gensym())
        push!(subs, newvar => ex)
        return newvar
    end
    @assert any(Base.Fix1(isequal,ex), exceptions) "The expression contains a time-dependent term $(ex) without explicit timepoints."
    if istree(ex)
        op = operation(ex)
        args = map(arguments(ex)) do u
            replace_timedeps!(u, subs; exceptions)
        end
        return op(args...)
    end
    # We check here if the nontree is a parameter
    return ex
end

function collect_timepoints(subs, tpoints=Float64[])
    for (k, v) in subs
        append!(tpoints, last(arguments(v)))
    end
    unique!(sort!(tpoints))
    return tpoints
end

function find_subexpressions(subs)
    subexpressions = []
    for (k, v) in subs
        args = arguments(v)
        ex = length(args) == 3 ? args[2] : args[1]
        push!(subexpressions, ex)
    end
    unique!(subexpressions)
    subreplace = [xi => Symbolics.variable(gensym()) for xi in subexpressions]
    map(subs) do (k, v)
        newval = substitute(v, subreplace)
        k => newval 
    end, subreplace
end

function expand_time_expressions(d)
    expr = []
    for (ex, v) in d
        xs = Symbolics.get_variables(ex)
        @info xs
    end
end



