# Piecewise constant control function
function find_index(timestops, t)
    idx = searchsortedlast(timestops, t)
    # We always assume that we have left / right continuity
    min(max(firstindex(timestops), idx), lastindex(timestops))
end

function δₜ(p, timestops, t)
    @assert size(timestops, 1) == size(p, 1) "The dimensionality of the provided `tstops` and `parameters` are not consistent."
    id = find_index(timestops, t)
    getindex(p, id)
end


@register_symbolic δₜ(p::AbstractVector, timestops::AbstractVector, t::Real)::Real


"""
$(TYPEDEF)

Extends the system using a `searchsortedlast` based approach to find the active local control at the specified timepoints for each control signal.

# Fields
$(FIELDS)

# Formulations

If the control signal `u` is a variable, the equation will be added directly

```julia
u ~ u_i[searchsortedlast(u_t, t)]
```

where `u_i` are the local controls and `u_t` the timepoints.
"""
struct SearchIndexControl{D} <: AbstractControlFormulation
    "The control specifications"
    controls::D
end

SearchIndexControl(x) = begin
    specs = (__preprocess_control_specs(x),)
    SearchIndexControl{typeof(specs)}(specs)
end

SearchIndexControl(args...) = begin
    specs = _preprocess_control_specs(args...)
    SearchIndexControl{typeof(specs)}(specs)
end

function expand_formulation(::SearchIndexControl, sys, spec::NamedTuple)
    (; variable, differential, bounds, timepoints, defaults) = spec
    new_parameters = []
    callback_eqs = []
    new_equations = Equation[]
    D = Differential(ModelingToolkit.independent_variable(sys))
    control_var = Num(ModelingToolkit.getvar(sys, Symbol(variable), namespace=false))
    local_controlsym = Symbol(variable, :ᵢ)
    timepoint_sym = Symbol(variable, :ₜ)
    N = length(timepoints)
    ps = @parameters begin
        ($local_controlsym)[1:N] = defaults, [bounds = bounds, localcontrol = true]
        ($timepoint_sym)[1:N] = timepoints, [tstop = true, tunable=false]
    end
    append!(new_parameters, ps[1:2])
    if !differential
        append!(
            new_equations, [
                control_var ~ δₜ(ps[1], ps[2], t)
            ]
        )
    else
        append!(
            new_equations, [
                D(control_var) ~ δₜ(ps[1], ps[2], t)
            ]
        )

    end
    controlsys = ODESystem(
        new_equations,
        ModelingToolkit.independent_variable(sys), [], new_parameters;
        name=nameof(sys),
        discrete_events=callback_eqs,
    )
    extend(sys, controlsys)
end
