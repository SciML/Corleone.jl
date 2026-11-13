import SymbolicIndexingInterface: is_parameter, is_variable

"""
$(TYPEDEF)

Metadata for a state or parameter definition, including its selector, default,
bounds, kind, and integrality. Concrete annotations are resolved against a symbolic
index provider before numerical layers are constructed.
"""
abstract type AbstractAnnotation end

"""
    Annotations(sys; kwargs...)

Read state and parameter metadata from a symbolic system. The ModelingToolkit
extension returns `(; variables, parameters)`, containing [`Tunable`](@ref) or
[`Constant`](@ref) annotations with explicit indices, defaults, bounds and kinds.
Load ModelingToolkit to use this constructor.
"""
function Annotations end

for f in (:index, :sym, :default, :kind, :integer, :transform, :prior)
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

get_bounds(x::AbstractAnnotation) = annotation_bounds(get_transform(x))
get_lowerbound(x::AbstractAnnotation) = first(get_bounds(x))
get_upperbound(x::AbstractAnnotation) = last(get_bounds(x))
has_lowerbound(x::AbstractAnnotation) = !isnothing(get_lowerbound(x))
has_upperbound(x::AbstractAnnotation) = !isnothing(get_upperbound(x))
set_lowerbound(x::AbstractAnnotation, value) = remake(x; lowerbound = value)
set_upperbound(x::AbstractAnnotation, value) = remake(x; upperbound = value)
has_bounds(x::AbstractAnnotation) = has_lowerbound(x) &&  has_upperbound(x)

function is_fully_specified(x::AbstractAnnotation)
    return is_minimum_specified(x) && has_index(x) && has_sym(x) && has_default(x) &&
        has_bounds(x) && has_integer(x)
end

function is_minimum_specified(x::AbstractAnnotation)
    return (has_index(x) || has_sym(x)) &&
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
A missing default takes the current value in `problem`; without a problem, it
falls back to the symbolic default of `indep`, then zero.
"""
function update_annotation(x::AbstractAnnotation, indep, problem = nothing)
    is_minimum_specified(x) || throw(
        ArgumentError(
            "Provide an index or symbol and a parameter or variable kind for $x."
        )
    )
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
    default = has_default(x) ? get_default(x) : fallback_default(problem, x, indep, sym, index)
    transform = resolve_transform(get_transform(x), get(default_bounds(indep), sym, (-Inf, Inf)))
    validate_transform(transform)
    lowerbound, upperbound = get_bounds(transform)
    lowerbound <= upperbound || throw(ArgumentError("Lower bound exceeds upper bound."))
    validate_annotation_default(default, transform)
    # Keep unbounded integer annotations representable; bounds need not have the
    # same type as the default. Explicit false must also override model metadata.
    integer = has_integer(x) ? get_integer(x) : is_integer_variable(sym)
    validate_prior(get_prior(x))
    return remake(x; index, sym, default, transform, integer)
end

# The problem holds the values a simulation would use; symbolic defaults may predate them.
fallback_default(::Nothing, x, indep, sym, index) = get(SymbolicIndexingInterface.default_values(indep), sym, 0.0)
function fallback_default(problem, x, indep, sym, index)
    index isa Integer || return fallback_default(nothing, x, indep, sym, index)
    source = is_parameter(x) ? problem.p : problem.u0
    values, _, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), source)
    return values[index]
end

# Prior support is supplied by the Distributions extension.
validate_prior(::Nothing) = nothing
validate_prior(prior) = throw(ArgumentError("Unsupported prior $(typeof(prior)); load Distributions for univariate priors."))

# Callbacks are checked after evaluation, without drawing from their RNG twice.
# Each natural value must lie in the transform's domain, so its internal value is finite.
validate_annotation_default(default, transform) = default
function validate_annotation_default(default::Number, transform)
    in_domain(transform, default) || throw(ArgumentError("Default value is outside the annotation bounds or transform domain."))
    return default
end
function validate_annotation_default(default::AbstractVector, transform)
    all(value -> value isa Number && in_domain(transform, value), default) ||
        throw(ArgumentError("Default values must be numbers within the annotation bounds or transform domain."))
    return default
end

"""
$(SIGNATURES)

Initialize `n` scalar values from a scalar, vector, or callback. Count callbacks
`(rng, n::Int)` take precedence, preserving untyped callbacks. Otherwise a callback
accepting `(rng, times::AbstractVector)` receives the supplied initialization times.
Validate the result's shape and the domain of `transform` once, without reevaluating
the callback.
Supplied vectors and callback results are retained by reference.
"""
function initialize_annotation_values(default, rng, n::Int, times, transform)
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
    return validate_annotation_default(values, transform)
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
    "Whether the value is integer-valued, or `nothing` to inherit symbol metadata."
    integer
    "A BoundedTransform identity or an explicit scalar transform from internal to natural values."
    transform
    "A univariate prior distribution on natural values, or `nothing`."
    prior
end

is_tunable(::DynamicVariable{T}) where {T} = T

# Preserve the tunability type parameter while allowing field types to change.
function SciMLBase.remake(
        x::DynamicVariable{T};
        index = x.index, sym = x.sym, kind = x.kind, default = x.default,
        lowerbound = missing, upperbound = missing, integer = x.integer,
        transform = x.transform, prior = x.prior,
    ) where {T}
    transform = annotation_transform(transform, nothing, nothing)
    transform = replace_transform_bounds(transform, lowerbound, upperbound)
    return DynamicVariable{T}(index, sym, kind, default, integer, transform, prior)
end

function Base.show(io::IO, ::MIME"text/plain", f::DynamicVariable{T}) where {T}
    name = f.sym isa AbstractVector ? string(f.sym) :
        has_sym(f) ? string(SymbolicIndexingInterface.getname(f.sym)) : ""
    output = isempty(name) ? "" : name * "="
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
    return print(io, output)
end

"""
$(SIGNATURES)

Construct an annotation whose values are stored in Lux parameters. Supply an
index or symbol, a vector of indices or symbols, a positional `selector => default` pair, or the `index`/`sym`
keywords. Optional keywords are `kind`, `default`, `lowerbound`, `upperbound`,
`integer`, `transform`, and `prior`. Missing metadata is resolved from the symbolic
container. For an [`EvaluationLayer`](@ref) keyword, supply a scalar `default`
and let the keyword provide the name.

`transform` is an increasing scalar transform from internal to natural values (load
TransformVariables, e.g. `asℝ₊`). Lux parameters then store internal values, while
defaults and layer outputs remain natural. With no explicit transform, a
[`BoundedTransform`](@ref) stores the bounds and uses identity coordinates. Explicit
transforms replace this object and supply their own optimizer bounds;
TransformVariables transforms use `(-Inf, Inf)`. `prior` is a univariate
distribution on natural values (load Distributions); see [`log_prior`](@ref).

Use `kind = Val(:variable)` for initial-condition resets and `Val(:parameter)`
for parameter definitions. Pair the annotation with its time grid to define a layer.

`default` accepts a scalar, a vector, or a callback `(rng, n::Int) -> values` or
`(rng, times::AbstractVector) -> values`. Untyped callbacks retain the count-based
interpretation; the count signature wins if both are applicable. State defaults
have one value per reset time, in supplied grid order. Parameter defaults have one
value per interval; time-based callbacks receive `[tspan[1]; switching_times]`.
A parameter without switching times has one value at `tspan[1]`.

For vector selectors, supply one default per selector: a vector of scalars gives
constant guesses, and a vector of vectors gives per-variable time series. Omitted
defaults inherit metadata separately. Bounds, transforms, and priors may be shared or
per-selector vectors. Grouped annotations expand into scalar definitions before layer construction.

Scalar and vector bounds are checked when resolving annotations. Callback output
length and bounds are checked during Lux setup. Vectors and callback results are
retained by reference; use a callback that allocates when independent setup arrays
are needed.
"""
function Tunable(;
        index = nothing, sym = nothing, kind::Val = Val{:unknown}(), default = nothing,
        lowerbound = nothing, upperbound = nothing, integer::Union{Nothing, Bool} = nothing,
        transform = nothing, prior = nothing
    )
    isnothing(index) && isnothing(sym) && isnothing(default) &&
        throw(ArgumentError("Provide an index, a symbol, or a default for a named evaluation parameter."))
    return DynamicVariable{true}(index, sym, kind, default, integer, annotation_transform(transform, lowerbound, upperbound), prior)
end

function Tunable(index_or_sym; kwargs...)
    return if SymbolicIndexingInterface.symbolic_type(index_or_sym) == SymbolicIndexingInterface.NotSymbolic()
        Tunable(; index = index_or_sym, kwargs...)
    else
        Tunable(; sym = index_or_sym, kwargs...)
    end
end

function Tunable(selectors::AbstractVector; kwargs...)
    return all(x -> x isa Integer, selectors) ?
        Tunable(; index = selectors, kwargs...) : Tunable(; sym = selectors, kwargs...)
end

Tunable(x::Pair; kwargs...) = Tunable(first(x); default = last(x), kwargs...)

"""
$(SIGNATURES)

Construct a fixed-value annotation with the same selectors and metadata keywords
as [`Tunable`](@ref). Values are stored in Lux state through [`FixedLayer`](@ref),
and active fixed initial conditions do not contribute shooting residuals. For an
[`EvaluationLayer`](@ref) keyword, supply a scalar `default` and let the keyword
provide the name.
"""
function Constant(;
        index = nothing, sym = nothing, kind::Val = Val{:unknown}(), default = nothing,
        lowerbound = nothing, upperbound = nothing, integer::Union{Nothing, Bool} = nothing,
        transform = nothing, prior = nothing
    )
    isnothing(index) && isnothing(sym) && isnothing(default) &&
        throw(ArgumentError("Provide an index, a symbol, or a default for a named evaluation parameter."))
    return DynamicVariable{false}(index, sym, kind, default, integer, annotation_transform(transform, lowerbound, upperbound), prior)
end

function Constant(index_or_sym; kwargs...)
    return if SymbolicIndexingInterface.symbolic_type(index_or_sym) == SymbolicIndexingInterface.NotSymbolic()
        Constant(; index = index_or_sym, kwargs...)
    else
        Constant(; sym = index_or_sym, kwargs...)
    end
end

function Constant(selectors::AbstractVector; kwargs...)
    return all(x -> x isa Integer, selectors) ?
        Constant(; index = selectors, kwargs...) : Constant(; sym = selectors, kwargs...)
end

Constant(x::Pair; kwargs...) = Constant(first(x); default = last(x), kwargs...)

function scalar_annotations(annotation::AbstractAnnotation)
    index, sym = get_index(annotation), get_sym(annotation)
    selectors = index isa AbstractVector ? index : sym
    selectors isa AbstractVector || return (annotation,)
    isempty(selectors) && throw(ArgumentError("An annotation must select at least one entry."))
    n = length(selectors)
    default = get_default(annotation)
    lowerbound, upperbound = get_bounds(annotation)
    transform, prior = get_transform(annotation), get_prior(annotation)
    for (field, value) in (
            (:index, index), (:sym, sym), (:default, default),
            (:lowerbound, lowerbound), (:upperbound, upperbound),
            (:transform, transform), (:prior, prior),
        )
        if field in (:index, :sym, :default) && !isnothing(value)
            value isa AbstractVector || throw(ArgumentError("Grouped annotation $field must be a vector."))
        end
        value isa AbstractVector && length(value) != n &&
            throw(DimensionMismatch("Grouped annotation $field must have $n entries."))
    end
    entry(value, i) = value isa AbstractVector ? value[i] : value
    return map(eachindex(selectors)) do i
        remake(
            annotation;
            index = entry(index, i), sym = entry(sym, i),
            default = entry(default, i),
            lowerbound = entry(lowerbound, i),
            upperbound = entry(upperbound, i),
            transform = entry(transform, i),
            prior = entry(prior, i)
        )
    end
end


process_definition(y, x::Val, ks::Tuple, v; problem = nothing) = reduce(
    vcat, map(ks) do k
        process_definition(y, x, k, v; problem)
    end
)

process_definition(y, x::Val, ks::AbstractArray, v; problem = nothing) = map(ks) do k
    process_definition(y, x, k, v; problem)
end

function process_definition(indep, kind::Val, k, v; problem = nothing)
    var = Tunable(k, kind = kind)
    return update_annotation(var, indep, problem) => v
end

function process_definition(indep, kind::Val, annotation::AbstractAnnotation, grid; problem = nothing)
    has_kind(annotation) && get_kind(annotation) != kind &&
        throw(ArgumentError("Annotation kind does not match the layer being constructed."))
    resolved = map(scalar_annotations(annotation)) do scalar
        update_annotation(set_kind(scalar, kind), indep, problem) => grid
    end
    return resolved isa Tuple ? only(resolved) : resolved
end
