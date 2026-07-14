using SymbolicIndexingInterface: to_dict_or_nothing

"""
$(TYPEDEF)

A `ControlSymbolCache` is a wrapper around a `SymbolCache` that allows for the inclusion of control symbols. It provides methods for querying the properties of the symbols, such as whether they are variables or parameters, and their indices.

Quadrature symbols are state variables that represent running cost integrals. They are kept at their natural variable indices and are tracked separately so that the segment hierarchy can exclude them from shooting continuity constraints and report cumulative totals across shooting segments.

# Fields
$(FIELDS)
"""
@concrete terse struct ControlSymbolCache
    "Original symbol cache"
    sys
    "Control symbols"
    controls
    "Parameter index map"
    parameter_indices
    "Quadrature symbols"
    quadratures
end

function ControlSymbolCache(sys, controls=[], quadratures=Symbol[])
    control_param_indices = filter!(!isnothing, map(Base.Fix1(parameter_index, sys), controls))
    parameter_indices = setdiff(parameter_symbols(sys), controls)
    var_offset = maximum(variable_symbols(sys); init=0) do v
        maximum(variable_index(sys, v); init=0)
    end
    control_var_indices = var_offset .+ (1:length(controls))
    controls = Dict(zip(controls, zip(control_var_indices, control_param_indices)))
    parameter_indices = Dict([sym => parameter_index(sys, sym) for sym in parameter_indices])
    quadratures = Set{Symbol}(quadratures)
    ControlSymbolCache(sys, controls, parameter_indices, quadratures)
end

subscript(i) = join(Char(0x2080 + digit) for digit in digits(i))

default_cache(prob::P where P <: SciMLBase.AbstractDEProblem) = SymbolCache(
    [Symbol(:u, subscript(i)) for i in eachindex(prob.u0)], 
    [Symbol(:p, subscript(i)) for i in eachindex(prob.p)],
    :t
)

get_symbolic_container(f) = begin 
    sys = symbolic_container(f)
    isnothing(sys) && return sys 
    isempty(variable_symbols(sys)) && return nothing 
    sys 
end

maybegetme(ps, index::Base.AbstractVecOrTuple) = map(Base.Fix1(maybegetme, ps), index)
maybegetme(ps, index) = begin 
    @assert index ∈ ps "$index not in collection!"
    index 
end

maybegetme(ps, index::Union{<:Int, UnitRange}) = ps[index]

function ControlSymbolCache(prob::P where P <: SciMLBase.AbstractDEProblem, controls::AbstractVector = [], quadratures::AbstractVector = []) 
    sys = something(get_symbolic_container(prob.f), default_cache(prob))
    @info sys
    ps = parameter_symbols(sys)
    sort!(ps, by = Base.Fix1(parameter_index, sys))
    us = variable_symbols(sys)
    sort!(us, by = Base.Fix1(variable_index, sys))
    controls = reduce(vcat, map(Base.Fix1(maybegetme, ps), controls), init = eltype(ps)[])
    quadratures = reduce(vcat, map(Base.Fix1(maybegetme, us), quadratures), init = eltype(us)[])
    return ControlSymbolCache(sys, controls, quadratures)
end

"""
Return the variable indices of all quadrature symbols in the underlying symbol cache.
"""
quadrature_indices(cache::ControlSymbolCache) =
    sort([variable_index(cache.sys, s) for s in cache.quadratures])

SymbolicIndexingInterface.is_independent_variable(sys::ControlSymbolCache, var) = is_independent_variable(sys.sys, var)

SymbolicIndexingInterface.independent_variable_symbols(sys::ControlSymbolCache) = independent_variable_symbols(sys.sys)

function SymbolicIndexingInterface.is_variable(cache::ControlSymbolCache, sym)
    haskey(cache.controls, sym) || is_variable(cache.sys, sym)
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
    union(variable_symbols(csys.sys), keys(csys.controls))
end

function SymbolicIndexingInterface.is_parameter(csys::ControlSymbolCache, sym)
    !haskey(csys.controls, sym) && is_parameter(csys.sys, sym)
end

function SymbolicIndexingInterface.parameter_index(csys::ControlSymbolCache, sym)
    if haskey(csys.controls, sym)
        return nothing
    else
        return get(csys.parameter_indices, sym, nothing)
    end
end

function SymbolicIndexingInterface.parameter_symbols(csys::ControlSymbolCache)
    setdiff(parameter_symbols(csys.sys), keys(csys.controls))
end

function SymbolicIndexingInterface.is_timeseries_parameter(csys::ControlSymbolCache, sym)
    is_parameter(csys, sym) && is_timeseries_parameter(csys.sys, sym)
end
