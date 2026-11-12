"""
$(TYPEDEF)

Wrap a SymbolicIndexingInterface symbolic container with bookkeeping metadata.
`controls` selects parameters; `quadratures` selects states that integrate
running costs. Selections accept symbols or original indices. All symbols retain
their original kind, index and ordering. Controls remain parameters.

`bounds` maps state or parameter symbols to `(lower, upper)` pairs. `integers`
lists the state and parameter symbols whose optimization values are integral. The
parent container and supplied bounds dictionary are retained by reference, just as
the parent's defaults are. Index-provider queries delegate through `symbolic_container`.
A differential equation problem can also be supplied in place of `sys`.

# Fields
$(FIELDS)
"""
@concrete terse struct ControlSymbolCache
    "The original symbolic index provider, retained by reference."
    sys
    "The subset of parameter symbols designated as controls."
    controls
    "The subset of state symbols designated as running-cost integrals."
    quadratures
    "State and parameter symbols mapped to `(lower, upper)` bounds, retained by reference."
    bounds
    "State and parameter symbols whose optimization values are integral."
    integers
end

function metadata_subset(sys, selected, kind)
    symbols, indexof = kind == :parameter ?
        (parameter_symbols(sys), parameter_index) : (variable_symbols(sys), variable_index)
    selectors = selected isa Union{AbstractArray, Tuple} ? selected : (selected,)
    subset = map(selectors) do selector
        position = findfirst(symbols) do sym
            selector isa Integer ? isequal(indexof(sys, sym), selector) : isequal(sym, selector)
        end
        isnothing(position) && throw(ArgumentError("Unknown $kind selection: $selector"))
        symbols[position]
    end
    allunique(subset) || throw(ArgumentError("Duplicate $kind selections."))
    return collect(eltype(symbols), subset)
end

"""
$(SIGNATURES)

Construct a bookkeeping wrapper from a symbolic index provider. `controls` and
`quadratures` accept symbols or original indices; `bounds` and `integers` must be
keyed by symbols.
The parent and bounds dictionary are retained by reference. Passing an existing
cache without overrides returns that same cache.
"""
function ControlSymbolCache(sys; controls = (), quadratures = (), bounds = default_bounds(sys), integers = default_integers(sys))
    controls = metadata_subset(sys, controls, :parameter)
    quadratures = metadata_subset(sys, quadratures, :variable)
    for (sym, interval) in bounds
        (is_variable(sys, sym) || is_parameter(sys, sym)) ||
            throw(ArgumentError("Bounds refer to an unknown state or parameter: $sym"))
        symbolic_type(sym) == NotSymbolic() &&
            throw(ArgumentError("Bounds must be keyed by symbols, not numeric indices."))
        interval isa Union{Tuple, AbstractVector} && length(interval) == 2 ||
            throw(ArgumentError("Bounds for $sym must contain a lower and upper bound."))
        first(interval) <= last(interval) || throw(ArgumentError("Lower bound exceeds upper bound for $sym."))
    end
    for sym in integers
        (is_variable(sys, sym) || is_parameter(sys, sym)) && symbolic_type(sym) != NotSymbolic() ||
            throw(ArgumentError("Integer selections must be state or parameter symbols, got $sym."))
    end
    return ControlSymbolCache(sys, controls, quadratures, bounds, collect(Any, integers))
end

ControlSymbolCache(prob::SciMLBase.AbstractSciMLProblem; kwargs...) =
    ControlSymbolCache(get_symbolic_container(prob); kwargs...)

function ControlSymbolCache(
        cache::ControlSymbolCache;
        controls = cache.controls, quadratures = cache.quadratures, bounds = cache.bounds,
        integers = cache.integers
    )
    controls === cache.controls && quadratures === cache.quadratures &&
        bounds === cache.bounds && integers === cache.integers && return cache
    return ControlSymbolCache(cache.sys; controls, quadratures, bounds, integers)
end

# SII already forwards its entire index-provider interface through this hook.
SymbolicIndexingInterface.symbolic_container(cache::ControlSymbolCache) = cache.sys
# Corleone's recursive problem lookup must retain the bookkeeping wrapper.
__get_symbolic_container(cache::ControlSymbolCache) = cache

default_bounds(cache::ControlSymbolCache) = cache.bounds
default_integers(cache::ControlSymbolCache) = cache.integers
default_integers(::Any) = ()
# These bookkeeping queries belong to this wrapper, not the SII interface.
"""
$(SIGNATURES)

Return the stored `(lower, upper)` bound pair for `sym`, or `(-Inf, Inf)`
when no bounds are stored for that symbol.
"""
get_bounds(cache::ControlSymbolCache, sym) = get(cache.bounds, sym, (-Inf, Inf))
"""
$(SIGNATURES)

Return whether `sym` is an integer state or parameter in `cache`. Symbols not
listed there fall back to their own metadata ([`is_integer_variable`](@ref)).
"""
is_integer_variable(cache::ControlSymbolCache, sym) =
    any(isequal(sym), cache.integers) || is_integer_variable(sym)
"""
$(SIGNATURES)

Return the parameter symbols designated as controls in `cache`.
This bookkeeping query is defined only for [`ControlSymbolCache`](@ref).
"""
control_symbols(cache::ControlSymbolCache) = cache.controls
"""
$(SIGNATURES)

Return the original parameter indices of the controls in `cache`.
This bookkeeping query is defined only for [`ControlSymbolCache`](@ref).
"""
control_indices(cache::ControlSymbolCache) = map(sym -> parameter_index(cache, sym), cache.controls)
"""
$(SIGNATURES)

Return the state symbols designated as quadratures in `cache`.
This bookkeeping query is defined only for [`ControlSymbolCache`](@ref).
"""
quadrature_symbols(cache::ControlSymbolCache) = cache.quadratures
"""
$(SIGNATURES)

Return the original state indices of the quadratures in `cache`.
This bookkeeping query is defined only for [`ControlSymbolCache`](@ref).
"""
quadrature_indices(cache::ControlSymbolCache) = map(sym -> variable_index(cache, sym), cache.quadratures)
