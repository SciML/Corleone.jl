"""
$(TYPEDEF)

Extends the system using a `ifelse` based approach to find the active local control at the specified timepoints for each control signal.

# Fields
$(FIELDS)

# Formulations

If the control signal `u` is a variable, the equation will be added directly

```julia
u ~ sum(ifelse((t >= uₜ[i]) && (uₜ[i+1] > t), uᵢ[i], 0) for i in eachindex(uᵢ))
```

where `u_i` are the local controls and `u_t` the timepoints.
"""
struct IfElseControl{D} <: AbstractControlFormulation
    "The control specifications"
    controls::D

    function IfElseControl(x...)
        specs = _preprocess_control_specs(x...)
        return new{typeof(specs)}(specs)
    end
end

function _expand_ifelse(t, ts, ps)
    eq = Num(0)
    for i in axes(ts, 1)
        eq += if i == lastindex(ts)
            ifelse(t >= ts[i], ps[i], 0)
        elseif i == firstindex(ts)
            ifelse(t < ts[i+1], ps[i], 0)
        else
            ifelse((t >= ts[i]) & (t < ts[i+1]), ps[i], 0)
        end
    end
    eq
end


function expand_formulation(::IfElseControl, sys, spec::NamedTuple)
    (; variable, differential, bounds, timepoints, independent_variable, defaults) = spec
    new_parameters = []
    callback_eqs = Equation[]
    new_equations = Equation[]
    D = Differential(ModelingToolkit.get_iv(sys))
    control_var = Num(ModelingToolkit.getvar(sys, Symbol(variable), namespace=false))
    iv = Num(ModelingToolkit.getvar(sys, Symbol(independent_variable), namespace=false))
    local_controlsym = Symbol(variable, :ᵢ)
    timepoint_sym = Symbol(variable, :ₜ)
    N = length(timepoints)
    ps = @parameters begin
        ($local_controlsym)[1:N] = defaults, [bounds = bounds, localcontrol = true]
        ($timepoint_sym)[1:N] = timepoints, [tstop = true, tunable = false]
    end
    append!(new_parameters, ps[1:2])
    if !differential
        append!(
            new_equations, [
                control_var ~ _expand_ifelse(iv, ps[2], ps[1])
            ]
        )
    else
        append!(
            new_equations, [
                D(control_var) ~ _expand_ifelse(iv, ps[2], ps[1])
            ]
        )

    end
    controlsys = System(
        new_equations,
        ModelingToolkit.get_iv(sys), [], new_parameters;
        name=nameof(sys),
        discrete_events=[timepoints => nothing],
    )
    extend_system(sys, controlsys)
end
