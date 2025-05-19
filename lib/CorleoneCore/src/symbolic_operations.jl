"""
$(TYPEDEF) 

Defines an integral over the provided expression with respect to the provided variable.
"""
struct Integral <: Symbolics.Operator 
    "The variable or expression to integrate over."
    x
    Integral(x) = new(x) 
    Integral(x::Union{AbstractFloat, Integer}) = error("Integral($x) is not a valid operator. Integrals should be taken w.r.t. a symbolic variable.")
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
        return x*f.x
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

"""
$(TYPEDEF)

Defines an operator to annotate an expression with the corresponding timepoints.
"""
struct ForAll{T<:Number} <: Symbolics.Operator
    "The timepoint for evaluation"
    timepoint::T
end

Base.:(==)(f::ForAll, g::ForAll) = isequal(f.timepoint, g.timepoint)
Base.hash(f::ForAll, salt::UInt) = Base.hash(f.timepoint, salt)


ForAll(ex, timepoint::Number) = ForAll(timepoint)(ex)
ForAll(ex::AbstractArray, timepoint::Number) = ForAll(timepoint).(ex)
ForAll(ex, timepoint::AbstractVector) = map(t -> ForAll(ex, t), timepoint)
∀(ex, timepoints) = ForAll(ex, timepoints)

has_forall(ex) = Symbolics.recursive_hasoperator(ForAll, ex)

SymbolicUtils.promote_symtype(::Type{ForAll}, T, args...) = T
SymbolicUtils.isbinop(::ForAll) = false
Base.nameof(::ForAll) = :ForAll
Base.show(io::IO, x::ForAll{<:Number}) = print(io, "(t = ", x.timepoint, ") : ")
ModelingToolkit.input_timedomain(::ForAll, _=nothing) = SciMLBase.ContinuousClock()
ModelingToolkit.output_timedomain(::ForAll, _=nothing) = SciMLBase.ContinuousClock()

function (f::ForAll)(x::Union{Equation,Inequality})
    xnew = Symbolics.canonical_form(x).lhs
    t = f(xnew)
    isa(x, Equation) ? t ~ 0 : t ≲ 0
end

function (f::ForAll{T})(x) where {T}
    iw = Symbolics.iswrapped(x)
    x = Symbolics.unwrap(x)
    if Symbolics.symbolic_type(x) == Symbolics.NotSymbolic()
        return x
    end
    result = if Symbolics.symbolic_type(x) == Symbolics.ArraySymbolic()
        Symbolics.array_term(f, x)
    else
        SymbolicUtils.term(f, x)
    end
    iw ? Symbolics.wrap(result) : result
end

function SymbolicUtils.maketerm(::Type{<:SymbolicUtils.BasicSymbolic}, t::ForAll, args, meta)
    val = t(args...)
    if symbolic_type(val) == NotSymbolic()
        return val
    end
    return SymbolicUtils.metadata(val, meta)
end


## Collect the corresponding variables and lower those
function __find_terms!(condition, subs, current)
    if condition(current)
        push!(subs, current)
    end
    istree(current) || return
    foreach(arguments(current)) do next
        __find_terms!(condition, subs, next)
    end
    return
end

function find_lagrangeterms!(subs, current)
    __find_terms!(is_lagrangeterm, subs, current)
end

function extend_intergral_terms(sys, exprs::Num...)
    lagrangeterms = []
    newstates = []
    neweqs = Equation[]
    subs = Pair[]
    foreach(exprs) do ex
        find_lagrangeterms!(lagrangeterms, ex)
    end
    unique!(lagrangeterms)
    foreach(enumerate(lagrangeterms)) do (i, t)
        diffeq, iv = arguments(Symbolics.unwrap(t))
        gradop = Differential(iv)
        varsym = Symbol(:𝕃, Symbol(Char(0x2080 + i)))
        newvar = @variables ($varsym)(iv) = 0.0
        append!(newstates, newvar)
        push!(neweqs, gradop(newvar[1]) ~ diffeq)
        push!(subs, t => newvar[1])
    end
    lagrange_sys = ODESystem(
        neweqs, independent_variable(sys), newstates, [],
        name=Symbol(:Integral, :_, nameof(sys))
    )
    newexprs = map(Base.Fix2(substitute, subs), exprs)
    extend(sys, lagrange_sys), newexprs
end

# We have a very specific case here. 
# If the objective is of the form min sum(args...) 
# and any of the args is equal to the iv of the sys, then 
# we transform the time
# If the objective is purely min iv(sys) also 
function change_of_variables(objective, sys, exprs...)
    iv = independent_variable(sys)
    obj = Symbolics.unwrap(objective)
    transform_iv = false
    if !istree(obj)
        transform_iv = isequal(obj, iv)
    else
        transform_iv = (operation(obj) ∈ (+, sum)) && any(Base.Fix1(isequal, iv), arguments(obj))
    end
    transform_iv || return (sys, exprs...)
    D = Differential(iv)
    varsym = Symbol(:τ, :ᵢ, :ᵥ)
    psym = Symbol(:T, :ᵢ, :ᵥ)
    p = only(@parameters ($psym) = 1.0 [bounds = (eps(), Inf)])
    x = only(@variables ($varsym)(iv))
    # TODO Double check here!
    eqs = Equation[
        D(x)~inv(p),
        iv~p*x
    ]
    paramsys = ODESystem(Equation[], iv, [], [p,], name=Symbol(:ChangeOfVariable, :_, nameof(sys)))
    sys = extend(sys, paramsys)

    new_objective = substitute(objective - iv + p, iv => x)
    new_exprs = map(Base.Fix2(substitute, iv => x), exprs)
    newsys = change_independent_variable(sys, x, eqs)
    newsys, new_objective
end

