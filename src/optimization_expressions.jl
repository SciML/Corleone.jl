"""
$(SIGNATURES)

Select `Val(:RuntimeGeneratedFunctions)` when its extension is loaded, otherwise
return `Val(:Eval)`. An explicit `evaluator` keyword overrides this selection.
"""
function default_evaluator()
    return isnothing(Base.get_extension(@__MODULE__, :CorleoneRuntimeGeneratedFunctionsExtension)) ?
        Val(:Eval) : Val(:RuntimeGeneratedFunctions)
end

"""
    compile_evaluator(strategy::Val, expression::Expr, eval_module::Module)

Compile a one-argument lambda expression into a numerical callable. `Val(:Eval)` uses
Core.eval in eval_module and calls the resulting function in its latest world.
The RuntimeGeneratedFunctions extension implements `Val(:RuntimeGeneratedFunctions)`.
Parsing and index substitution are shared by both implementations.
"""
function compile_evaluator end

call_eval_function(f, x) = Base.invokelatest(f, x)
compile_evaluator(::Val{:Eval}, expression::Expr, eval_module::Module) =
    Base.Fix1(call_eval_function, Core.eval(eval_module, expression))

function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(call_eval_function), f, x)
    y, pullback = Base.invokelatest(rrule_via_ad, config, f, x)
    function eval_pullback(dy)
        df, dx = Base.invokelatest(pullback, dy)
        return NoTangent(), df, dx
    end
    return y, eval_pullback
end

"""
$(TYPEDEF)

A lazy scalar-indexed view of saved states or right-continuous parameters.
The kind selects independent state/parameter indexing; both retain original SII
indices. No shared row numbering or state-count offset is introduced.

# Fields
$(FIELDS)
"""
struct ExpressionSamples{T, K}
    "The trajectory supplying current numerical values and saved times."
    trajectory::T
end
Base.getindex(samples::ExpressionSamples{T, :states}, j::Int, i::Int) where {T} =
    state_values(samples.trajectory)[i][j]
function Base.getindex(samples::ExpressionSamples{T, :parameters}, j::Int, i::Int) where {T}
    traj = samples.trajectory
    p = SymbolicIndexingInterface.parameter_values_at_time(traj, traj, current_time(traj)[i])
    return p[j]
end
Base.getindex(samples::ExpressionSamples, j::Int) =
    map(i -> samples[j, i], eachindex(current_time(samples.trajectory)))

expression_symbol_name(sym::Symbol) = sym
expression_symbol_name(sym) = throw(ArgumentError("Expression parsing is not implemented for $(typeof(sym))."))

function expression_symbol_indices(layer)
    indices = Dict{Symbol, Tuple{Symbol, Int}}()
    for (kind, symbols, indexof) in (
            (:states, variable_symbols(layer.sys), variable_index),
            (:parameters, parameter_symbols(layer.sys), parameter_index),
        )
        for sym in symbols
            name = expression_symbol_name(sym)
            haskey(indices, name) && throw(ArgumentError("State and parameter names must be distinct in timed expressions."))
            indices[name] = (kind, indexof(layer.sys, sym))
        end
    end
    return indices
end

function expression_sample_lookup(input, location, indices...)
    kind, j = location
    source = Expr(:., input, QuoteNode(kind))
    return Expr(:call, GlobalRef(Base, :getindex), source, j, indices...)
end

function expression_time(x)
    if x isa Real && !(x isa Bool)
        isfinite(x) || throw(ArgumentError("Expression times must be finite."))
        return x
    end
    throw(ArgumentError("Signal times must be finite real literals; interpolate computed times into the Expr. Got $x."))
end

function collect_expression_times!(timepoints, expr::Expr, indices)
    if expr.head === :call
        op = first(expr.args)
        if op isa Symbol && haskey(indices, op)
            length(expr.args) == 2 || throw(ArgumentError("A signal call must have exactly one time argument: $expr."))
            push!(get!(timepoints, op, Real[]), expression_time(expr.args[2]))
            return timepoints
        end
    end
    for arg in expr.args
        collect_expression_times!(timepoints, arg, indices)
    end
    return timepoints
end
function collect_expression_times!(timepoints, sym::Symbol, indices)
    haskey(indices, sym) && get!(timepoints, sym, Real[])
    return timepoints
end
collect_expression_times!(timepoints, x, indices) = timepoints

function expression_evaluation_grid(layer, st, timepoints)
    extra = Real[t for times in values(timepoints) for t in times]
    lo, hi = layer.problem.tspan
    all(t -> lo <= t <= hi, extra) || throw(DomainError(extra, "Expression times must lie within the problem time span."))
    isempty(extra) && return st.timestops
    T = promote_type(eltype(st.timestops), map(typeof, extra)...)
    return sort!(unique!(vcat(collect(T, st.timestops), collect(T, extra))))
end

function replace_expression_samples(expr::Expr, indices, times, input)
    if expr.head === :call && first(expr.args) isa Symbol && haskey(indices, first(expr.args))
        symbol = first(expr.args)
        t = convert(eltype(times), expression_time(expr.args[2]))
        i = searchsortedlast(times, t) # Select the outgoing state at resets.
        i > 0 && times[i] == t || throw(ArgumentError("Expression time $t was not saved by the solver."))
        return expression_sample_lookup(input, indices[symbol], i)
    end
    # Function names are not signal values. Preserve ordinary Julia calls, while
    # replacing their arguments; other expression heads retain their Julia meaning.
    args = expr.head === :call ?
        vcat(Any[first(expr.args)], map(x -> replace_expression_samples(x, indices, times, input), expr.args[2:end])) :
        map(x -> replace_expression_samples(x, indices, times, input), expr.args)
    return Expr(expr.head, args...)
end
function replace_expression_samples(sym::Symbol, indices, times, input)
    return haskey(indices, sym) ? expression_sample_lookup(input, indices[sym]) : sym
end
replace_expression_samples(x, indices, times, input) = x

function compile_trajectory_expression(expr::Expr, indices, times, eval_module, evaluator)
    input = gensym(:samples)
    body = replace_expression_samples(expr, indices, times, input)
    f = compile_evaluator(evaluator, Expr(:->, input, body), eval_module)
    return traj -> f(
        (;
            states = ExpressionSamples{typeof(traj), :states}(traj),
            parameters = ExpressionSamples{typeof(traj), :parameters}(traj),
        )
    )
end
