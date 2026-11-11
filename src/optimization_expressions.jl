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

expression_evaluation_grid(layer, st, timepoints) = expression_sample_grid(layer.problem.tspan, st.timestops, values(timepoints))

function expression_sample_grid((lo, hi), grid, timepoints)
    extra = Real[t for times in timepoints for t in times]
    all(t -> lo <= t <= hi, extra) || throw(DomainError(extra, "Expression times must lie within the problem time span."))
    isempty(extra) && return grid
    T = promote_type(eltype(grid), map(typeof, extra)...)
    return sort!(unique!(vcat(collect(T, grid), collect(T, extra))))
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

"""
$(TYPEDEF)

A compiled trajectory expression retaining parsed signal dependencies for structural
Jacobian preparation. Literal vector expressions additionally retain dependencies
per component; other vector-valued expressions use their combined dependencies.

# Fields
$(FIELDS)
"""
struct ParsedTrajectoryExpression{F, D, C}
    "The numerical evaluator accepting state and parameter sample views."
    f::F
    "Referenced (kind, index, time) signals; nothing denotes the entire series."
    dependencies::D
    "Per-component dependencies for a literal vector, or nothing."
    components::C
end
function (expression::ParsedTrajectoryExpression)(traj)
    return expression.f(
        (;
            states = ExpressionSamples{typeof(traj), :states}(traj),
            parameters = ExpressionSamples{typeof(traj), :parameters}(traj),
        )
    )
end

function collect_expression_dependencies!(dependencies, expr, indices, times)
    if expr isa Symbol
        haskey(indices, expr) && push!(dependencies, (indices[expr]..., nothing))
    elseif expr isa Expr
        if expr.head === :call && first(expr.args) isa Symbol && haskey(indices, first(expr.args))
            push!(dependencies, (indices[first(expr.args)]..., convert(eltype(times), expression_time(expr.args[2]))))
        else
            args = expr.head === :call ? expr.args[2:end] : expr.args
            foreach(arg -> collect_expression_dependencies!(dependencies, arg, indices, times), args)
        end
    end
    return dependencies
end
function expression_dependencies(expr, indices, times)
    dependencies = Tuple{Symbol, Int, Union{Nothing, Real}}[]
    return unique!(collect_expression_dependencies!(dependencies, expr, indices, times))
end

function compile_trajectory_expression(expr::Expr, indices, times, eval_module, evaluator)
    input = gensym(:samples)
    body = replace_expression_samples(expr, indices, times, input)
    f = compile_evaluator(evaluator, Expr(:->, input, body), eval_module)
    components = if expr.head === :vect && !any(arg -> arg isa Expr && arg.head === :..., expr.args)
        map(arg -> expression_dependencies(arg, indices, times), expr.args)
    else
        nothing
    end
    dependencies = isnothing(components) ? expression_dependencies(expr, indices, times) :
        unique!(reduce(vcat, components; init = Tuple{Symbol, Int, Union{Nothing, Real}}[]))
    return ParsedTrajectoryExpression(f, dependencies, components)
end

# Opaque callbacks may read any trajectory value. Parsed expressions expose the
# signals they read; unknown vector output structure shares their union per row.
trajectory_constraint_dependencies(f, n, resolve, ncolumns) =
    fill(BitSet(1:ncolumns), n)
function trajectory_constraint_dependencies(f::ParsedTrajectoryExpression, n, resolve, ncolumns)
    if !isnothing(f.components) && length(f.components) == n
        return map(resolve, f.components)
    end
    columns = resolve(f.dependencies)
    return fill(columns, n)
end
