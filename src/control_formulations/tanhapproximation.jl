"""
$(TYPEDEF)

Extends the system using `tanh` as a smooth approximation for the heaviside step function.

# Fields
$(FIELDS)

# Formulations

If the control signal `u` is a variable, the equation will be added directly

```julia
```

where `u_i` are the local controls and `u_t` the timepoints.

# Note

This formulation is differentiable with respect to the timepoints.
"""
struct TanhControl{D} <: AbstractControlFormulation
    "The control specifications"
    controls::D
    TanhControl(args...) = begin
        specs = _preprocess_control_specs(args...)
        new{typeof(specs)}(specs)
    end
end

__tanh(x, k) = 1 / 2 * (1 + tanh(k * x))

function _expand_tanh(t, k, ts, ps)
    eq = Num(0)
    for i in axes(ts, 1)
        eq += if i == lastindex(ts)
            __tanh(t - ts[i], k) * ps[i]
        elseif i == firstindex(ts)
            (1 - __tanh(t - ts[i+1], k)) * ps[i]
        else
            __tanh(t - ts[i], k) * (1 - __tanh(t - ts[i+1], k)) * ps[i]
        end
    end
    simplify(eq)
end


function expand_formulation(::TanhControl, sys, spec::NamedTuple)
    (; variable, differential, timepoints, independent_variable, defaults) = spec
    new_parameters = []
    callback_eqs = []
    new_equations = Equation[]
    D = Differential(ModelingToolkit.get_iv(sys))
    control_var = Num(ModelingToolkit.getvar(sys, Symbol(variable), namespace=false))
    iv = Num(ModelingToolkit.getvar(sys, Symbol(independent_variable), namespace=false))
    local_controlsym = Symbol(variable, :ᵢ)
    timepoint_sym = Symbol(variable, :ₜ)
    transition_sym = Symbol(variable, :ₖ)
    N = length(timepoints)
    lower, upper = ModelingToolkit.getbounds(control_var)
    ps = @parameters begin
        ($local_controlsym)[1:N] = defaults, [bounds = (lower, upper), localcontrol = true,
                differentialcontrol = differential]
        ($timepoint_sym)[1:N] = timepoints, [tunable = false, tstop = true]
        ($transition_sym) = 50.0, [tunable = false, bounds = (0.0, Inf)]
    end
    append!(new_parameters, ps)
    if !differential
        append!(
            new_equations, [
                control_var ~ _expand_tanh(iv, ps[3], ps[2], ps[1])
            ]
        )
    else
        append!(
            new_equations, [
                D(control_var) ~ _expand_tanh(iv, ps[3], ps[2], ps[1])
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
