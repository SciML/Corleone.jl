"""
$(TYPEDEF)

Extends the system using a `DiscreteCallback` at the specified timepoints for each control signal. This callback increments an integer variable at all specific timepoints.

# Fields
$(FIELDS)

# Formulations

If the control signal `u` is a variable, the corresponding callback will be added directly

```math
u = u_i[k]
```

If the control signal acts on a differential variable, we use a dummy parameter `uₚ`

```math
u_t = u_p,
u_p = u_i[k]
```
"""
struct IndexControlCallback{D} <: AbstractControlFormulation
    "The control specifications"
    controls::D
end

# TODO This does not work due to either system initialization or callbacks.
# Maybe switch to imperative callbacks here.
function (method::IndexControlCallback)(sys)
    throw(error("$(typeof(method)) is currently not implemented. Please switch to a different formulation."))
    control_specs = get_control_specs(method)
    new_parameters = []
    callback_eqs = []
    new_equations = Equation[]
    initialization_eqs = Equation[]
    t = independent_variable(sys) |> Num
    D = Differential(t)
    for (k, timepoints) in control_specs
        N = length(timepoints)
        ui = Symbolics.unwrap(k)
        lower, upper = getbounds(ui)
        isdifferential = false
        usym = if iscall(ui) && isa(operation(ui), Symbolics.Differential)
            isdifferential = true
            defval = [0.0 for _ in Base.OneTo(N)]
            operation(only(arguments(ui)))
        elseif iscall(ui)
            defval = [Symbolics.getdefaultval(ui) for _ in Base.OneTo(N)]
            operation(ui)
        else
            defval = [Symbolics.getdefaultval(ui) for _ in Base.OneTo(N)]
            ui
        end
        localsym = Symbol(usym, :ᵢ)
        tswitches = Symbol(usym, :ₜ)
        indexsym = Symbol(usym, :ₖ)
        changepoint = Symbol(usym, :ₛ)
        ps = @parameters begin
            ($localsym)[1:N] = defval, [bounds = (lower, upper), localcontrol = true,
                    differentialcontrol = differential]
            ($tswitches)[1:N] = timepoints, [tstop = true]
            ($indexsym)(t)::Int = 1
        end
        append!(new_parameters, ps)
        if !isdifferential
                push!(callback_eqs,  timepoints => [ps[3] ~ ps[3] + 1])
            push!(
                new_equations, k ~ ps[1][ps[3]]
            )
        end
    end
    controlsys = ODESystem(
        new_equations, t, [], new_parameters;
        name=Symbol(:DirectControlCallback, :_, nameof(sys)),
        discrete_events=callback_eqs,
        initialization_eqs=initialization_eqs
    )
    extend(sys, controlsys)
end
