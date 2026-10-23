"""
$(TYPEDEF)

A `ControlSymbolCache` is a wrapper around a `SymbolCache` that allows for the inclusion of control symbols. It provides methods for querying the properties of the symbols, such as whether they are variables or parameters, and their indices.

Quadrature symbols are state variables that represent running cost integrals. They are kept at their natural variable indices and tracked separately so that they can be reported as cumulative totals across piecewise segments and ensemble shooting intervals.

# Fields
$(FIELDS)
"""
@concrete terse struct ControlSymbolCache
    "Symbolic cache of the underlying differential equation"
    sys
    "Non-quadrature state symbols mapped to their original state indices"
    variables
    "Control symbols mapped to appended state and original parameter indices"
    controls
    "Quadrature symbols mapped to their original state indices"
    quadratures
    "Ordinary parameter symbols mapped to their original parameter indices"
    parameters
    "Bounds for the variables"
    bounds
end

"""$(SIGNATURES)

Build a control-aware cache from a symbolic container and selected controls and quadratures.
"""
function ControlSymbolCache(
        sys; controls = [], quadratures = [],
        bounds = Dict(), kwargs...
    )
    control_param_indices = filter!(!isnothing, map(Base.Fix1(parameter_index, sys), controls))
    parameter_indices = setdiff(parameter_symbols(sys), controls)
    var_offset = maximum(variable_symbols(sys); init = 0) do v
        maximum(variable_index(sys, v); init = 0)
    end
    control_var_indices = var_offset .+ (1:length(controls))

    control_indices = OrderedDict(zip(controls, zip(control_var_indices, control_param_indices)))
    parameter_indices = OrderedDict([sym => parameter_index(sys, sym) for sym in parameter_indices])
    quadrature_indices = OrderedDict([sym => variable_index(sys, sym) for sym in quadratures])
    variable_indices = OrderedDict([sym => variable_index(sys, sym) for sym in setdiff(variable_symbols(sys), quadratures)])
    return ControlSymbolCache(sys, variable_indices, control_indices, quadrature_indices, parameter_indices, bounds)
end

"""$(SIGNATURES)

Return the symbols treated as controls.
"""
control_symbols(csys::ControlSymbolCache) = keys(csys.controls)
"""$(SIGNATURES)

Return the symbols treated as quadrature states.
"""
quadrature_symbols(csys::ControlSymbolCache) = keys(csys.quadratures)
"""$(SIGNATURES)

Return the state symbols excluding quadratures and promoted controls.
"""
minimal_variable_symbols(csys::ControlSymbolCache) = keys(csys.variables)
SymbolicIndexingInterface.parameter_symbols(csys::ControlSymbolCache) = keys(csys.parameters)

@non_differentiable control_symbols(sys)
@non_differentiable quadrature_symbols(sys)
@non_differentiable minimal_variable_symbols(sys)

"""$(SIGNATURES)

Return appended state indices for controls.
"""
control_indices(csys::ControlSymbolCache) = first.(values(csys.controls)) |> collect
original_control_indices(cache::ControlSymbolCache) = last.(values(cache.controls)) |> collect
"""$(SIGNATURES)

Return the original state indices of quadratures.
"""
quadrature_indices(csys::ControlSymbolCache) = values(csys.quadratures) |> collect
"""$(SIGNATURES)

Return the original state indices of non-quadrature variables.
"""
minimal_variable_indices(csys::ControlSymbolCache) = values(csys.variables) |> collect

__state_permutation(cache::ControlSymbolCache) = vcat(
    minimal_variable_indices(cache), quadrature_indices(cache), control_indices(cache)
)

@non_differentiable control_indices(sys)
@non_differentiable original_control_indices(sys)
@non_differentiable quadrature_indices(sys)
@non_differentiable minimal_variable_indices(sys)
@non_differentiable __state_permutation(sys)


subscript(i) = begin
    join(Char(0x2080 + digit) for digit in reverse(digits(i)))
end

function default_cache(prob::P where {P <: SciMLBase.AbstractDEProblem})
    defaults = Dict{Symbol, Base.promote_eltype(prob.u0, prob.p)}()
    varsyms = Symbol[]
    for (i, u0) in enumerate(prob.u0)
        varsym = Symbol(:x, subscript(i))
        defaults[varsym] = u0
        push!(varsyms, varsym)
    end
    psyms = []
    for (i, p0) in enumerate(prob.p)
        psym = Symbol(:p, subscript(i))
        defaults[psym] = p0
        push!(psyms, psym)
    end
    tsym = :t
    return SymbolCache(
        varsyms, psyms, tsym; defaults
    )
end

get_symbolic_container(prob::P where {P <: SciMLBase.AbstractDEProblem}) = begin
    sys = symbolic_container(prob.f)
    if typeof(sys) <: SymbolCache{Nothing, Nothing, Nothing, Nothing, Dict{Any, Any}}
        return default_cache(prob)
    end
    return sys
end

maybegetme(ps, index::Base.AbstractVecOrTuple) = map(Base.Fix1(maybegetme, ps), index)
maybegetme(ps, index) = begin
    @assert index ∈ ps "$index not in collection!"
    index
end
maybegetme(ps, index::Union{<:Int, UnitRange}) = ps[index]

"""$(SIGNATURES)

Build a cache for a differential equation problem; controls and quadratures may be symbols or indices.
"""
function ControlSymbolCache(
        prob::P where {P <: SciMLBase.AbstractDEProblem};
        controls::AbstractVector = [],
        quadratures::AbstractVector = [],
        kwargs...
    )
    sys = get_symbolic_container(prob)
    ps = parameter_symbols(sys)
    sort!(ps, by = Base.Fix1(parameter_index, sys))
    us = variable_symbols(sys)
    sort!(us, by = Base.Fix1(variable_index, sys))
    controls = reduce(vcat, map(Base.Fix1(maybegetme, ps), controls), init = eltype(ps)[])
    quadratures = reduce(vcat, map(Base.Fix1(maybegetme, us), quadratures), init = eltype(us)[])
    return ControlSymbolCache(sys; controls, quadratures)
end

SymbolicIndexingInterface.is_independent_variable(sys::ControlSymbolCache, var) = is_independent_variable(sys.sys, var)

SymbolicIndexingInterface.independent_variable_symbols(sys::ControlSymbolCache) = independent_variable_symbols(sys.sys)

function SymbolicIndexingInterface.is_variable(cache::ControlSymbolCache, sym)
    return haskey(cache.controls, sym) || is_variable(cache.sys, sym)
end

SymbolicIndexingInterface.constant_structure(cache::ControlSymbolCache) = constant_structure(cache.sys)

SymbolicIndexingInterface.all_variable_symbols(cache::ControlSymbolCache) = union(all_variable_symbols(cache.sys), keys(cache.controls))

SymbolicIndexingInterface.all_symbols(cache::ControlSymbolCache) = all_symbols(cache.sys)

SymbolicIndexingInterface.default_values(cache::ControlSymbolCache) =
    default_values(cache.sys)

SymbolicIndexingInterface.is_time_dependent(cache::ControlSymbolCache) = is_time_dependent(cache.sys)

function SymbolicIndexingInterface.variable_index(cache::ControlSymbolCache, sym)
    if haskey(cache.controls, sym)
        return cache.controls[sym][1]
    else
        return variable_index(cache.sys, sym)
    end
end

function SymbolicIndexingInterface.variable_symbols(csys::ControlSymbolCache)
    return union(variable_symbols(csys.sys), keys(csys.controls))
end

function SymbolicIndexingInterface.is_parameter(csys::ControlSymbolCache, sym)
    return !haskey(csys.controls, sym) && is_parameter(csys.sys, sym)
end

function SymbolicIndexingInterface.parameter_index(csys::ControlSymbolCache, sym)
    if haskey(csys.controls, sym)
        return nothing
    else
        return get(csys.parameters, sym, nothing)
    end
end

function SymbolicIndexingInterface.is_timeseries_parameter(csys::ControlSymbolCache, sym)
    return is_parameter(csys, sym) && is_timeseries_parameter(csys.sys, sym)
end

bounds(csys::ControlSymbolCache, var) = get(csys.bounds, var, (-Inf, Inf))
upperbounds(csys::ControlSymbolCache, var) = map(last, bounds(csys, var))
lowerbounds(csys::ControlSymbolCache, var) = map(first, bounds(csys, var))
