function system_annotation(sys, sym, kind, values, defaults; tunable)
    index = kind === Val(:variable) ? variable_index(sys, sym) : parameter_index(sys, sym)
    index isa Integer || throw(ArgumentError("MTK annotations require scalar signals and flat parameter indices; compile with split=false."))
    default = isnothing(values) ? get(defaults, sym, Symbolics.getdefaultval(sym, nothing)) : values[index]
    isnothing(default) && (default = Symbolics.substitute(sym, defaults))
    default = SymbolicUtils.unwrap_const(Symbolics.unwrap(default))
    default isa Real || throw(ArgumentError("Supply a numerical default for $sym through the system or u0/p."))
    lower, upper = map(bound -> SymbolicUtils.unwrap_const(Symbolics.unwrap(bound)), ModelingToolkit.getbounds(sym))
    constructor = tunable ? Tunable : Constant
    return constructor(;
        index, sym, kind, default, lowerbound = lower, upperbound = upper,
        integer = SymbolicUtils.symtype(sym) <: Integer
    )
end

"""
$(SIGNATURES)

Parse ModelingToolkit state and parameter metadata into
`(; variables, parameters)` collections of Corleone annotations. Each annotation
has an explicit scalar index, symbol, kind, numerical default, bounds and
integrality. Array components become scalar annotations in column-major order.
Tunability follows MTK metadata. `u0` and `p`, when supplied, override
symbolic defaults in numerical index order, preserving problem initial values.

`controls` selects parameters that must be tunable regardless of their metadata.
`lift_states = true` makes all state annotations tunable for full-state shooting
resets. Parameter indices must be flat; use `mtkcompile(...; split = false)`.
"""
function Corleone.Annotations(
        sys::ModelingToolkit.System; u0 = nothing, p = nothing,
        controls = (), lift_states::Bool = false
    )
    sys = scalarize_system(sys)
    defaults = default_values(sys)
    variables = map(variable_symbols(sys)) do sym
        system_annotation(
            sys, sym, Val(:variable), u0, defaults;
            tunable = lift_states || ModelingToolkit.istunable(sym, false)
        )
    end
    parameters = map(parameter_symbols(sys)) do sym
        system_annotation(
            sys, sym, Val(:parameter), p, defaults;
            tunable = any(input -> isequal(Symbolics.unwrap(input), sym), controls) || ModelingToolkit.istunable(sym)
        )
    end
    return (; variables, parameters)
end
