"""
$(TYPEDEF)

Abstract type defining different formulation for piecewise constant control signals.

All discrete implementations take in a list of variables and time points, which are in turn extended.
"""
abstract type AbstractControlFormulation end

(x::AbstractControlFormulation)(sys) = expand_formulation(x, sys)

function expand_formulation(x::AbstractControlFormulation, sys)
    newsys = expand_formulation(x, sys, x.controls...)
    newsys = @set newsys.costs = ModelingToolkit.get_costs(sys)
    newsys = @set newsys.consolidate = ModelingToolkit.get_consolidate(sys)
    newsys = @set newsys.tspan = ModelingToolkit.get_tspan(sys)
end

function expand_formulation(x::AbstractControlFormulation, sys, specs...)
    sys = expand_formulation(x, sys, first(specs))
    expand_formulation(x, sys, Base.tail(specs)...)
end

function expand_formulation(x::AbstractControlFormulation, sys, ::NamedTuple)
    error("$(x) does not implement a concrete method to extend the system.")
end

"""
$(FUNCTIONNAME)

Returns the control specification, which should be a `Dict` of variables and time points.
"""
get_control_specs(x::AbstractControlFormulation) = x.controls

function __preprocess_control_specs(x::Any)
    throw(ArgumentError("The current control specification ($x) is not implemented. Please refer to the documentation on how to provide control specifications."))
end

function __preprocess_control_specs(nt::NamedTuple)
    @assert haskey(nt, :variable) "The provided specification $(nt) does not have the required key `variable`"
    @assert haskey(nt, :timepoints) "The provided specification $(nt) does not have the required key `timepoints`"
    (; variable, timepoints) = nt
    defaults = haskey(nt, :defaults) ? nt.defaults : nothing
    __preprocess_control_specs(variable, timepoints, defaults)
end

function __preprocess_control_specs(spec::Pair{Num,<:Base.AbstractVecOrTuple})
    __preprocess_control_specs((; variable=first(spec), timepoints=collect(last(spec))))
end

function __preprocess_control_specs(spec::Pair{Num,<:NamedTuple})
    __preprocess_control_specs(merge(last(spec), (; variable=first(spec))))
end

function __preprocess_control_specs(variable, timepoints, defaults=nothing)
    variable = Symbolics.unwrap(first(variable))
    differential = false
    if Symbolics.iscall(variable) && isa(operation(variable), Differential)
        variable = first(arguments(variable))
        differential = true
    end
    if isnothing(defaults) && Symbolics.hasmetadata(variable, Symbolics.VariableDefaultValue)
        defaults = [Symbolics.getdefaultval(variable) for _ in eachindex(timepoints)]
    elseif isnothing(defaults)
        throw(error("Control variable $(variable) does not have a default value and no defaults have been specified"))
    end
    bounds = getbounds(variable)
    @assert iscall(variable) "The control variables must be specified using x(t) or similar syntax."
    independent_variable = only(arguments(variable))
    #independent_variable = arguments(variable)
    #independent_variable = length(independent_variable) > 1 ? only(arguments(independent_variable[1])) : only(independent_variable)
    #variable = operation(variable) === getindex ? variable : operation(variable)
    variable = operation(variable)
    @assert size(timepoints) == size(defaults)
    idx = sortperm(timepoints)
    (; variable, differential, bounds, independent_variable, timepoints=getindex(timepoints, idx), defaults=getindex(defaults, idx))
end

function _preprocess_control_specs(specs::Base.Pair...)
    __preprocess_control_specs.(specs)
end

_preprocess_control_specs(spec::Base.Pair) = (__preprocess_control_specs(spec),)