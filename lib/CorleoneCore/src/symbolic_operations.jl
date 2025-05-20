# TODO Maybe go for Symbolics.Integral here
"""
$(TYPEDEF) 

Defines an integral over the provided expression with respect to the provided variable.
"""
struct Integral <: Symbolics.Operator
    "The variable or expression to integrate over."
    x
    Integral(x) = new(x)
    Integral(x::Union{AbstractFloat,Integer}) = error("Integral($x) is not a valid operator. Integrals should be taken w.r.t. a symbolic variable.")
end

const I_nounits = Integral(ModelingToolkit.t_nounits)
∫(x) = I_nounits(x)
∫(x, t) = Integral(t)(x)

Base.:(==)(a::Integral, b::Integral) = isequal(a.x, b.x)
Base.hash(a::Integral, salt::UInt) = hash(a.x, salt)

SymbolicUtils.promote_symtype(::Integral, T, args...) = T
SymbolicUtils.isbinop(::Integral) = false
Base.nameof(::Integral) = :Integral
Base.show(io::IO, x::Integral) = print(io, "∫(d", x.x, ")")
ModelingToolkit.input_timedomain(::Integral, _=nothing) = SciMLBase.ContinuousClock()
ModelingToolkit.output_timedomain(::Integral, _=nothing) = SciMLBase.ContinuousClock()

has_integrals(ex) = Symbolics.recursive_hasoperator(Integral, ex)

function (f::Integral)(x)
    iw = Symbolics.iswrapped(x)
    x = Symbolics.unwrap(x)
    if Symbolics.symbolic_type(x) == Symbolics.NotSymbolic()
        return x * f.x
    end
    result = if Symbolics.symbolic_type(x) == Symbolics.ArraySymbolic()
        Symbolics.array_term(f, x)
    elseif Symbolics.iscall(x) && operation(x) == Differential(f.x)
        only(arguments(x))
    else
        SymbolicUtils.term(f, x)
    end
    iw ? Symbolics.wrap(result) : result
end


function SymbolicUtils.maketerm(::Type{<:SymbolicUtils.BasicSymbolic}, t::Integral, args, meta)
    val = t(args...)
    if symbolic_type(val) == NotSymbolic()
        return val
    end
    return SymbolicUtils.metadata(val, meta)
end
