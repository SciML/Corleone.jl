# Only expression names are lowered; controls remain parameters.
Corleone.expression_symbol_name(sym::Union{Symbolics.Num, SymbolicUtils.BasicSymbolic}) = Symbol(sym)

# MTK's non-split problem constructor needs scalar parameter entries. Keep the
# indexed symbols and their metadata, but make the numerical ordering explicit.
scalar_symbols(symbols) = reduce(
    vcat, map(sym -> symbolic_type(sym) isa ArraySymbolic ? vec(collect(Symbolics.scalarize(sym))) : [sym], symbols);
    init = eltype(symbols)[]
)

scalar_expression(expression) = Symbolics.scalarize(expression)
# Symbolics.scalarize preserves array inequalities as a single relationship.
scalar_expression(inequality::Symbolics.Inequality) = Symbolics.Inequality.(
    Symbolics.scalarize(inequality.lhs), Symbolics.scalarize(inequality.rhs), Ref(inequality.relational_op)
)

scalar_expressions(expressions) = reduce(
    vcat, map(expressions) do expression
        scalar = scalar_expression(expression)
        scalar isa AbstractArray ? vec(collect(scalar)) : [scalar]
    end; init = eltype(expressions)[]
)

function scalarize_system(sys::ModelingToolkit.System)
    variables, parameters = variable_symbols(sys), parameter_symbols(sys)
    costs = scalar_expressions(ModelingToolkit.get_costs(sys))
    constraints = scalar_expressions(ModelingToolkit.get_constraints(sys))
    arrays = any(sym -> symbolic_type(sym) isa ArraySymbolic, (variables..., parameters...))
    !arrays && isequal(costs, ModelingToolkit.get_costs(sys)) &&
        isequal(constraints, ModelingToolkit.get_constraints(sys)) && return sys
    scalar = ModelingToolkit.System(
        ModelingToolkit.equations(sys), ModelingToolkit.get_iv(sys),
        scalar_symbols(variables), scalar_symbols(parameters);
        name = nameof(sys),
        initial_conditions = ModelingToolkit.initial_conditions(sys),
        bindings = ModelingToolkit.bindings(sys), guesses = ModelingToolkit.guesses(sys),
        observed = ModelingToolkit.observed(sys),
        costs, constraints,
        continuous_events = ModelingToolkit.continuous_events(sys),
        discrete_events = ModelingToolkit.discrete_events(sys),
        tstops = ModelingToolkit.symbolic_tstops(sys), checks = false
    )
    return ModelingToolkit.complete(scalar; split = false, add_initial_parameters = false)
end

function lower_integrals!(accumulations, ex, iv)
    ex = Symbolics.unwrap(ex)
    SymbolicUtils.iscall(ex) || return ex
    op, args = SymbolicUtils.operation(ex), SymbolicUtils.arguments(ex)
    if op isa Symbolics.Integral
        isequal(Symbolics.unwrap(op.domain.variables), Symbolics.unwrap(iv)) || throw(ArgumentError("Only integrals over the system time variable are supported."))
        q = get!(accumulations, ex) do
            name = gensym(:quadrature)
            Symbolics.unwrap(only(ModelingToolkit.@variables ($name)(iv) = 0.0 [tunable = false]))
        end
        lo, hi = op.domain.domain.left, op.domain.domain.right
        return SymbolicUtils.operation(q)(hi) - SymbolicUtils.operation(q)(lo)
    end
    return op(map(arg -> lower_integrals!(accumulations, arg, iv), args)...)
end

function symbolic_expression(ex, signals, terminal_time, times)
    ex = Symbolics.unwrap(ex)
    if haskey(signals, ex)
        name, kind = signals[ex]
        kind === :parameter || throw(ArgumentError("Specify a numerical time for $ex in a cost or constraint."))
        return Expr(:call, name, terminal_time)
    end
    if SymbolicUtils.iscall(ex)
        op, args = SymbolicUtils.operation(ex), SymbolicUtils.arguments(ex)
        for (signal, (name, _)) in signals
            if SymbolicUtils.iscall(signal) && isequal(op, SymbolicUtils.operation(signal))
                length(args) == 1 || throw(ArgumentError("Expected a scalar signal with one time argument: $ex."))
                time = SymbolicUtils.unwrap_const(Symbolics.unwrap(only(args)))
                time isa Real && isfinite(time) || throw(ArgumentError("Signal times must be finite real numbers: $ex."))
                push!(times, time)
                return Expr(:call, name, time)
            end
        end
        op isa SymbolicUtils.BasicSymbolic && throw(ArgumentError("Unknown or eliminated signal in $ex."))
        return Expr(:call, op, map(arg -> symbolic_expression(arg, signals, terminal_time, times), args)...)
    end
    value = SymbolicUtils.unwrap_const(ex)
    value isa Real && return value
    throw(ArgumentError("Unsupported symbolic expression: $ex."))
end

# Even a constant symbolic cost needs the core's expression preparation path.
expression_wrapper(ex) = ex isa Expr ? ex : Expr(:call, identity, ex)
