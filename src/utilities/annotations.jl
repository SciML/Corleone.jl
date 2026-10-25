import SymbolicIndexingInterface: is_parameter, is_variable

"""
$(TYPEDEF)

Metadata for a state or parameter definition, including its selector, default,
bounds, kind, and integrality. Concrete annotations are resolved against a symbolic
index provider before numerical layers are constructed.
"""
abstract type AbstractAnnotation end

for f in (:index, :sym, :default, :upperbound, :lowerbound, :kind, :integer)
    # Construct names dynamically (e.g., get_index, set_index!)
    getter = Symbol(:get_, f)
    checker = Symbol(:has_, f)
    setter = Symbol(:set_, f)

    @eval begin
        $(getter)(x::AbstractAnnotation) = getfield(x, $(QuoteNode(f)))
        $(setter)(x::AbstractAnnotation, val) = remake(x; $(f) = val)
    end
    if f != :kind
        @eval $(checker)(x::AbstractAnnotation) = !isnothing($(getter)(x))
    else
        @eval $(checker)(x::AbstractAnnotation) = $(getter)(x) != Val{:unknown}()
    end
end

set_parameter(x::AbstractAnnotation) = remake(x, kind = Val{:parameter}())
is_parameter(x::AbstractAnnotation) = has_kind(x) && get_kind(x) == Val{:parameter}()
set_variable(x::AbstractAnnotation) = remake(x, kind = Val{:variable}())
is_variable(x::AbstractAnnotation) = has_kind(x) && get_kind(x) == Val{:variable}()

is_integer_variable(x::AbstractAnnotation) = something(get_integer(x), false)
relax_integer_variable(x::AbstractAnnotation) = set_integer(x, false)

get_bounds(x::AbstractAnnotation) = (get_lowerbound(x), get_upperbound(x))
has_bounds(x::AbstractAnnotation) = has_lowerbound(x) &&  has_upperbound(x)

function is_fully_specified(x::AbstractAnnotation)
    return is_minimum_specified(x) && has_index(x) && has_sym(x) && has_default(x) &&
        has_bounds(x) && has_integer(x)
end

function is_minimum_specified(x::AbstractAnnotation)
    (has_index(x) || has_sym(x)) &&
        (is_parameter(x) || is_variable(x))
end

"""
$(SIGNATURES)

Return the dictionary of default `(lower, upper)` bounds keyed by symbols.
The fallback returns an empty dictionary; [`ControlSymbolCache`](@ref) returns
its stored bounds dictionary by reference.
"""
default_bounds(::Any) = Dict()
is_integer_variable(::Any) = false

"""
$(SIGNATURES)

Resolve an annotation against symbolic index provider `indep`. Fill missing
index, symbol, default, bounds, and integrality metadata. Reject unknown selectors,
inconsistent symbol/index pairs, reversed bounds, and defaults outside bounds.
"""
function update_annotation(x::AbstractAnnotation, indep)
    is_minimum_specified(x) || throw(ArgumentError(
        "Provide an index or symbol and a parameter or variable kind for $x."
    ))
    indexof = is_parameter(x) ? SymbolicIndexingInterface.parameter_index :
        SymbolicIndexingInterface.variable_index
    symbols = is_parameter(x) ? SymbolicIndexingInterface.parameter_symbols(indep) :
        SymbolicIndexingInterface.variable_symbols(indep)
    if has_index(x)
        index = get_index(x)
        # Symbol enumeration need not be in index order (e.g. parameter dictionaries).
        position = findfirst(sym -> isequal(indexof(indep, sym), index), symbols)
        isnothing(position) && throw(ArgumentError("Unknown annotation index: $index"))
        sym = symbols[position]
        if has_sym(x) && !isequal(indexof(indep, get_sym(x)), index)
            throw(ArgumentError("Annotation symbol and index refer to different entries."))
        end
    else
        sym = get_sym(x)
        index = indexof(indep, sym)
        isnothing(index) && throw(ArgumentError("Unknown annotation symbol: $sym"))
    end
    default = has_default(x) ? get_default(x) :
        get(SymbolicIndexingInterface.default_values(indep), sym, 0.0)
    fallback_lower, fallback_upper = get(default_bounds(indep), sym, (-Inf, Inf))
    lowerbound = has_lowerbound(x) ? get_lowerbound(x) : fallback_lower
    upperbound = has_upperbound(x) ? get_upperbound(x) : fallback_upper
    lowerbound <= upperbound || throw(ArgumentError("Lower bound exceeds upper bound."))
    validate_annotation_default(default, lowerbound, upperbound)
    # Keep unbounded integer annotations representable; bounds need not have the
    # same type as the default. Explicit false must also override model metadata.
    integer = has_integer(x) ? get_integer(x) : is_integer_variable(sym)
    return remake(x; index, sym, default, lowerbound, upperbound, integer)
end

# Callbacks are checked after evaluation, without drawing from their RNG twice.
validate_annotation_default(default, lower, upper) = default
function validate_annotation_default(default::Number, lower, upper)
    lower <= default <= upper || throw(ArgumentError("Default value is outside the annotation bounds."))
    return default
end
function validate_annotation_default(default::AbstractVector, lower, upper)
    all(value -> value isa Number && lower <= value <= upper, default) ||
        throw(ArgumentError("Default values must be numbers within the annotation bounds."))
    return default
end

"""
$(SIGNATURES)

Initialize `n` scalar values from a scalar, vector, or callback. Count callbacks
`(rng, n::Int)` take precedence, preserving untyped callbacks. Otherwise a callback
accepting `(rng, times::AbstractVector)` receives the supplied initialization times.
Validate the result's shape and bounds once, without reevaluating the callback.
Supplied vectors and callback results are retained by reference.
"""
function initialize_annotation_values(default, rng, n::Int, times, bounds)
    values = if default isa Number
        fill(default, n)
    elseif default isa AbstractVector
        default
    elseif applicable(default, rng, n)
        default(rng, n)
    elseif times isa AbstractVector && applicable(default, rng, times)
        default(rng, times)
    else
        throw(ArgumentError("Default must be a scalar, a vector, or an initializer accepting (rng, n::Int) or (rng, times::AbstractVector). Time-based initializers require initialization times."))
    end
    values isa AbstractVector || throw(ArgumentError("An initializer must return an AbstractVector."))
    length(values) == n || throw(DimensionMismatch("Expected $n initial values, got $(length(values))."))
    return validate_annotation_default(values, bounds...)
end

"""
$(TYPEDEF)

A state or parameter annotation. The type parameter `T` selects tunable (`true`)
or fixed (`false`) values. Missing metadata is represented by `nothing`, except
for the unknown kind `Val(:unknown)`. Use [`Tunable`](@ref) or [`Constant`](@ref)
to construct annotations.

# Fields
$(FIELDS)
"""
@concrete terse struct DynamicVariable{T} <: AbstractAnnotation
    "The original state or parameter index, or `nothing` before resolution."
    index
    "The state or parameter symbol, or `nothing` before resolution."
    sym
    "The annotation kind: `Val(:variable)`, `Val(:parameter)`, or `Val(:unknown)`."
    kind
    "A scalar, vector, count/timepoint initializer, or `nothing` to inherit the symbolic default."
    default
    "The lower bound, or `nothing` to inherit the symbolic bound."
    lowerbound
    "The upper bound, or `nothing` to inherit the symbolic bound."
    upperbound
    "Whether the value is integer-valued, or `nothing` to inherit symbol metadata."
    integer
end

is_tunable(::DynamicVariable{T}) where {T} = T

# Preserve the tunability type parameter while allowing field types to change.
function SciMLBase.remake(x::DynamicVariable{T};
        index = x.index, sym = x.sym, kind = x.kind, default = x.default,
        lowerbound = x.lowerbound, upperbound = x.upperbound, integer = x.integer,
    ) where {T}
    return DynamicVariable{T}(index, sym, kind, default, lowerbound, upperbound, integer)
end

function Base.show(io::IO, ::MIME"text/plain", f::DynamicVariable{T}) where T
    output = has_sym(f) ? string(SymbolicIndexingInterface.getname(f.sym)) * "=" : ""
    if has_default(f)
        output *= string(get_default(f))
    end
    output *= T ? " (free" : " (fixed"
    if is_parameter(f)
        output *= " parameter)"
    elseif is_variable(f)
        output *= " variable)"
    else
        output *= ")"
    end
    print(io, output)
end

"""
$(SIGNATURES)

Construct an annotation whose values are stored in Lux parameters. Supply an
index or symbol, a positional `selector => default` pair, or the `index`/`sym`
keywords. Optional keywords are `kind`, `default`, `lowerbound`, `upperbound`,
and `integer`. Missing metadata is resolved from the symbolic container.

Use `kind = Val(:variable)` for initial-condition resets and `Val(:parameter)`
for parameter definitions. Pair the annotation with its time grid to define a layer.

`default` accepts a scalar, a vector, or a callback `(rng, n::Int) -> values` or
`(rng, times::AbstractVector) -> values`. Untyped callbacks retain the count-based
interpretation; the count signature wins if both are applicable. State defaults
have one value per reset time, in supplied grid order. Parameter defaults have one
value per interval; time-based callbacks receive `[tspan[1]; switching_times]`.
A parameter without switching times has one value at `tspan[1]`.

Scalar and vector bounds are checked when resolving annotations. Callback output
length and bounds are checked during Lux setup. Vectors and callback results are
retained by reference; use a callback that allocates when independent setup arrays
are needed.
"""
function Tunable(; index = nothing, sym = nothing, kind::Val = Val{:unknown}(), default = nothing, lowerbound = nothing, upperbound = nothing, integer::Union{Nothing, Bool} = nothing)
    isnothing(index) && isnothing(sym) &&
        throw(ArgumentError("Provide either an index or a symbol."))
    DynamicVariable{true}(index, sym, kind, default, lowerbound,  upperbound, integer)
end

function Tunable(index_or_sym; kwargs...)
    if SymbolicIndexingInterface.symbolic_type(index_or_sym) == SymbolicIndexingInterface.NotSymbolic()
        Tunable(; index = index_or_sym, kwargs...)
    else
        Tunable(; sym = index_or_sym, kwargs...)
    end
end

Tunable(x::Pair; kwargs...) = Tunable(first(x); default = last(x), kwargs...)

"""
$(SIGNATURES)

Construct a fixed-value annotation with the same selectors and metadata keywords
as [`Tunable`](@ref). Values are stored in Lux state through [`FixedLayer`](@ref),
and active fixed initial conditions do not contribute shooting residuals.
"""
function Constant(; index = nothing, sym = nothing, kind::Val = Val{:unknown}(), default = nothing, lowerbound = nothing, upperbound = nothing, integer::Union{Nothing, Bool} = nothing)
    isnothing(index) && isnothing(sym) &&
        throw(ArgumentError("Provide either an index or a symbol."))
    DynamicVariable{false}(index, sym, kind, default, lowerbound,  upperbound, integer)
end

function Constant(index_or_sym; kwargs...)
    if SymbolicIndexingInterface.symbolic_type(index_or_sym) == SymbolicIndexingInterface.NotSymbolic()
        Constant(; index = index_or_sym, kwargs...)
    else
        Constant(; sym = index_or_sym, kwargs...)
    end
end

Constant(x::Pair; kwargs...) = Constant(first(x); default = last(x), kwargs...)

process_definition(y, x::Val, ks::Tuple, v) = reduce(
    vcat, map(ks) do k
        process_definition(y, x, k, v)
    end
)

process_definition(y, x::Val, ks::AbstractArray, v)  = map(ks) do k
    process_definition(y, x, k, v)
end

function process_definition(indep, kind::Val, k, v)
    var = Tunable(k,  kind = kind)
    update_annotation(var, indep) => v
end

function process_definition(indep, kind::Val, annotation::AbstractAnnotation, grid)
    has_kind(annotation) && get_kind(annotation) != kind &&
        throw(ArgumentError("Annotation kind does not match the layer being constructed."))
    return update_annotation(set_kind(annotation, kind), indep) => grid
end
