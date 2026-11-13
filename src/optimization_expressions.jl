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

# Evaluated lambdas receive a fresh closure type on every `Core.eval`. Reuse the
# function compiled for an identical expression and module, so rebuilt problems
# retain their types and all code specialized on them.
const EVAL_EXPRESSION_CACHE = Dict{Tuple{Module, Expr}, Any}()
const EVAL_EXPRESSION_LOCK = ReentrantLock()
function compile_evaluator(::Val{:Eval}, expression::Expr, eval_module::Module)
    keyexpr = Base.remove_linenums!(Base.deepcopy(expression))
    key = (eval_module, keyexpr)
    f = lock(EVAL_EXPRESSION_LOCK) do
        length(EVAL_EXPRESSION_CACHE) > 10_000 && empty!(EVAL_EXPRESSION_CACHE)
        get!(EVAL_EXPRESSION_CACHE, key) do
            Core.eval(eval_module, keyexpr)
        end
    end
    return Base.Fix1(call_eval_function, f)
end

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

expression_symbol_indices(layer) = signal_symbol_indices(layer.sys)

function signal_symbol_indices(sys)
    indices = Dict{Symbol, Tuple{Symbol, Int}}()
    for (kind, symbols, indexof) in (
            (:states, variable_symbols(sys), variable_index),
            (:parameters, parameter_symbols(sys), parameter_index),
        )
        for sym in symbols
            name = expression_symbol_name(sym)
            haskey(indices, name) && throw(ArgumentError("State and parameter names must be distinct in timed expressions."))
            indices[name] = (kind, indexof(sys, sym))
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
            first(indices[op]) === :evaluation && throw(ArgumentError("Evaluation parameter $op is scalar. Use $op without a time argument."))
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
    haskey(indices, sym) && first(indices[sym]) !== :evaluation && get!(timepoints, sym, Real[])
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
per component. Elementwise series expressions (see [`elementwise_series_expression`](@ref))
depend per output row on one saved sample of each series plus their timed signals.
Other vector-valued expressions use their combined dependencies.

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
    "Whether output row `k` reads only saved sample `k` of each referenced series."
    elementwise::Bool
end
ParsedTrajectoryExpression(f, dependencies, components) =
    ParsedTrajectoryExpression(f, dependencies, components, false)
function (expression::ParsedTrajectoryExpression)(traj, evaluation = ())
    return expression.f(
        (;
            states = ExpressionSamples{typeof(traj), :states}(traj),
            parameters = ExpressionSamples{typeof(traj), :parameters}(traj),
            evaluation,
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

# The generated argument name is fixed rather than a gensym: compiling the same
# expression on the same saved grid must produce the same function type, so
# rebuilt problems reuse derivative preparation and other specialized code.
# The `#` prefix cannot be written as an ordinary user identifier.
const EXPRESSION_SAMPLES_ARGUMENT = Symbol("##corleone_samples")

struct ObservationExpression{F, D}
    f::F
    dependencies::D
end

(expression::ObservationExpression)(u, p, t, evaluation = ()) =
    expression.f((; states = u, parameters = p, time = (t,), evaluation))

function observe(expression::ObservationExpression, traj, k, evaluation)
    t = traj.t[k]
    parameters = SymbolicIndexingInterface.parameter_values_at_time(traj, traj, t)
    return expression(traj.u[k], parameters, t, evaluation)
end

function observation_symbol_indices(sys, layer)
    indices = signal_symbol_indices(sys)
    for symbol in independent_variable_symbols(sys)
        name = expression_symbol_name(symbol)
        haskey(indices, name) && throw(ArgumentError("Independent variables and signals must have distinct names."))
        indices[name] = (:time, 1)
    end
    return add_evaluation_symbols!(indices, layer)
end

"""
    compile_observation_expression(expression, sys, times, layer, eval_module, evaluator)

Compile an observation `expression` in current-time signal names of `sys`, such as
`:(w * u₂)`, into a callable `f(u, p, t, evaluation = ())` that evaluates it at one
state, parameter vector, and time. `layer` supplies the symbols of an
[`EvaluationLayer`](@ref) (or `nothing`); pass [`evaluation_parameters`](@ref) as
`evaluation`. `times` is the saved grid, and timed reads such as `u₁(0.5)` are
rejected. [`expression_signals`](@ref) returns the signals the result reads.
"""
function compile_observation_expression(expression, sys, times, layer, eval_module, evaluator)
    indices = observation_symbol_indices(sys, layer)
    timepoints = Dict{Symbol, Vector{Real}}()
    collect_expression_times!(timepoints, expression, indices)
    all(isempty, values(timepoints)) ||
        throw(ArgumentError("Observation expressions use current-time signal names; sampling times belong in the observation grid."))
    input = EXPRESSION_SAMPLES_ARGUMENT
    body = Base.remove_linenums!(replace_expression_samples(expression, indices, times, input))
    return ObservationExpression(
        compile_evaluator(evaluator, Expr(:->, input, body), eval_module),
        expression_dependencies(expression, indices, times)
    )
end
@non_differentiable compile_observation_expression(::Any, ::Any, ::Any, ::Any, ::Any, ::Any)

function compile_trajectory_expression(expr::Union{Expr, Symbol}, indices, times, eval_module, evaluator)
    input = EXPRESSION_SAMPLES_ARGUMENT
    # Source locations would otherwise distinguish identical expressions.
    body = Base.remove_linenums!(replace_expression_samples(expr, indices, times, input))
    f = compile_evaluator(evaluator, Expr(:->, input, body), eval_module)
    components = if expr isa Expr && expr.head === :vect && !any(arg -> arg isa Expr && arg.head === :..., expr.args)
        map(arg -> expression_dependencies(arg, indices, times), expr.args)
    else
        nothing
    end
    dependencies = isnothing(components) ? expression_dependencies(expr, indices, times) :
        unique!(reduce(vcat, components; init = Tuple{Symbol, Int, Union{Nothing, Real}}[]))
    elementwise = isnothing(components) && elementwise_series_expression(expr, indices) === :series
    return ParsedTrajectoryExpression(f, dependencies, components, elementwise)
end

"""
    elementwise_series_expression(expr, indices) -> :series, :scalar, or nothing

Classify an expression syntactically for per-sample series sparsity. `:series`
means output element `k` reads only saved sample `k` of every referenced series
signal (plus scalar timed signals); `:scalar` means a timed signal or a constant;
`nothing` means neither can be guaranteed.

The rule is deliberately conservative. An expression is elementwise if it is

- a bare series signal such as `u₁` or `p₂`;
- `identity(x)` of an elementwise expression `x`;
- a dotted call `f.(args...)` or dotted operator such as `a .* b`, `.-a` or
  `a .<= b`, whose arguments are all elementwise or scalar, with at least one
  series argument.

Scalar arguments are timed signals such as `u₁(2.0)`, numeric literals,
interpolated numeric values or arrays, and dotted calls of scalar arguments only. Everything else (for example `sum(u₁)`,
`cumsum(u₁)`, undotted calls such as `u₁ * 2`, indexing, vector literals, global
names, and `@.`/`.=` forms) is not elementwise and keeps the union of all its
dependencies in every row. Row sparsity is additionally applied only when the
output length equals the number of saved samples.
"""
function elementwise_series_expression(expr, indices)
    if expr isa Symbol
        haskey(indices, expr) || return nothing
        return first(indices[expr]) === :evaluation ? :scalar : :series
    end
    expr isa Expr || return expr isa Union{Number, AbstractArray{<:Number}} ? :scalar : nothing
    if expr.head === :call
        op = first(expr.args)
        args = expr.args[2:end]
        if op isa Symbol && haskey(indices, op)
            return length(args) == 1 && args[1] isa Real ? :scalar : nothing # Timed signal.
        elseif op === :identity && length(args) == 1
            return elementwise_series_expression(only(args), indices) === :series ? :series : nothing
        elseif dotted_operator(op)
            return elementwise_broadcast(args, indices)
        end
        return nothing
    elseif expr.head === :. && length(expr.args) == 2 && expr.args[2] isa Expr && expr.args[2].head === :tuple
        # `f.(args...)` broadcasts f; f itself must not be a signal.
        expr.args[1] isa Symbol && !haskey(indices, expr.args[1]) || return nothing
        return elementwise_broadcast(expr.args[2].args, indices)
    end
    return nothing
end
function elementwise_broadcast(args, indices)
    any(arg -> arg isa Expr && arg.head in (:..., :kw, :parameters), args) && return nothing
    kinds = map(arg -> elementwise_series_expression(arg, indices), args)
    any(isnothing, kinds) && return nothing
    return any(==(:series), kinds) ? :series : :scalar
end
function dotted_operator(op)
    op isa Symbol || return false
    name = String(op)
    return length(name) > 1 && startswith(name, '.') && Base.isoperator(Symbol(name[nextind(name, 1):end]))
end

"""
    expression_signals(f)

Return the `(kind, index, time)` signals read by a compiled trajectory or observation
expression, or `nothing` for an opaque callback. `kind` is `:states`, `:parameters`,
`:time`, or `:evaluation`; `time === nothing` denotes a read of the entire series
(or, for observations, of the current sample). Resolve them to columns with the
`resolve` function of [`shooting_dependency_context`](@ref).
"""
expression_signals(@nospecialize(f)) = nothing
expression_signals(f::Union{ParsedTrajectoryExpression, ObservationExpression}) = f.dependencies

# Opaque callbacks may read any trajectory value. Parsed expressions expose the
# signals they read; unknown vector output structure shares their union per row.
# `resolve(signals, sample)` resolves series signals at one saved sample index,
# or their whole history when `sample` is nothing.
"""
    trajectory_constraint_dependencies(f, n, resolve, columns, nsamples)

Return `n` dependency rows (`BitSet`s of column ids) for the constraint or objective
function `f` with `n` outputs on a trajectory with `nsamples` saved samples.
Parsed expressions resolve their signals with `resolve` from
[`shooting_dependency_context`](@ref). Opaque callbacks may read any value they can
access: every row then holds `columns`, either a `BitSet` or a column count `ncolumns`
for `1:ncolumns`.
"""
trajectory_constraint_dependencies(@nospecialize(f), n, resolve, columns, nsamples) =
    fill(opaque_columns(columns), n)
opaque_columns(ncolumns::Integer) = BitSet(1:ncolumns)
opaque_columns(columns::BitSet) = columns
function trajectory_constraint_dependencies(@nospecialize(f::ParsedTrajectoryExpression), n, resolve, ncolumns, nsamples)
    if !isnothing(f.components) && length(f.components) == n
        return map(signals -> resolve(signals, nothing), f.components)
    end
    if f.elementwise && n == nsamples
        series = filter(signal -> signal[1] !== :evaluation && isnothing(signal[3]), f.dependencies)
        timed = filter(signal -> signal[1] === :evaluation || !isnothing(signal[3]), f.dependencies)
        shared = resolve(timed, nothing)
        return [union(shared, resolve(series, k)) for k in 1:n]
    end
    columns = resolve(f.dependencies, nothing)
    return fill(columns, n)
end
