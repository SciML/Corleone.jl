"""
$(TYPEDEF)

Extends the system using a `DiscreteCallback` at the specified timepoints for each control signal.

# Fields
$(FIELDS)

# Formulations

If the control signal `u` is a variable, the corresponding callback will be added directly

```math
u = u_i
```

If the control signal acts on a differential variable, we use a dummy parameter `uₚ`

```math
u_t = u_p,
u_p = u_i
```
"""
struct DirectControlCallback{D} <: AbstractControlFormulation
    "The control specifications"
    controls::D

    function DirectControlCallback(x...)
        specs = _preprocess_control_specs(x...)
        return new{typeof(specs)}(specs)
    end
end

function expand_formulation(::DirectControlCallback, sys, spec::NamedTuple)
    (; variable, differential, bounds, independent_variable, timepoints, defaults) = spec
    new_parameters = []
    callback_eqs = []
    new_equations = Equation[]
    iv = Num(ModelingToolkit.getvar(sys, Symbol(independent_variable), namespace=false))
    D = Differential(ModelingToolkit.get_iv(sys))
    control_var = Num(ModelingToolkit.getvar(sys, Symbol(variable), namespace=false))
    local_controlsym = Symbol(variable, :ᵢ)
    timepoint_sym = Symbol(variable, :ₜ)
    psym = Symbol(variable, :ₚ)
    N = length(timepoints)
    ps = @parameters begin
        ($local_controlsym)[1:N] = defaults, [bounds = bounds, localcontrol = true]
        ($timepoint_sym)[1:N] = timepoints, [tstop = true, tunable = false]
        ($psym) = 0.0
    end
    if !differential
        append!(new_parameters, ps[1:2])
        for i in eachindex(timepoints)
            push!(callback_eqs, (iv == ps[2][i]) => [control_var ~ ModelingToolkit.Pre(ps[1][i])])
        end
        append!(
            new_equations, [
                D(control_var) ~ 0
            ]
        )
    else
        append!(new_parameters, ps)
        for i in eachindex(timepoints)
            push!(callback_eqs, (iv == ps[2][i]) => [ps[3] ~ ModelingToolkit.Pre(ps[1][i])])
        end
        append!(
            new_equations, [
                D(control_var) ~ ps[3],
            ]
        )
    end
    controlsys = System(
        new_equations,
        ModelingToolkit.get_iv(sys), [], new_parameters;
        name=nameof(sys),
        discrete_events=callback_eqs,
    )
    extend(sys, controlsys)
end
