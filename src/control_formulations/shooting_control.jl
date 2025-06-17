"""
$(TYPEDEF)

Extends the system using classical shooting approach where the initial value of each interval is set.
# Fields
$(FIELDS)

# Formulations

If the control signal `u` is a variable, the equation will be added directly

```julia
```

where `u_i` are the local controls and `u_t` the timepoints.
"""
struct ShootingControl{D} <: AbstractControlFormulation
    "The control specifications"
    controls::D
end

ShootingControl(x) = begin
    specs = _preprocess_control_specs((x,)...)
    ShootingControl{typeof(specs)}(specs)
end

ShootingControl(args...) = begin
    specs = _preprocess_control_specs(args...)
    ShootingControl{typeof(specs)}(specs)
end

function expand_formulation(::ShootingControl, sys, spec::NamedTuple)
    (; variable, differential, bounds,  timepoints, independent_variable, defaults) = spec
    new_parameters = []
    callback_eqs = []
    new_equations = Equation[]
    init_equations = Equation[]
    D = Differential(ModelingToolkit.independent_variable(sys))
    control_var = Num(ModelingToolkit.getvar(sys, Symbol(variable), namespace=false))
    iv = Num(ModelingToolkit.getvar(sys, Symbol(independent_variable), namespace=false))
    local_controlsym = Symbol(variable, :ᵢ)
    timepoint_sym = Symbol(variable, :ₜ)
    u0_sym = Symbol(variable, Symbol(Char(0x2080)))
    N = length(timepoints)
    ps = @parameters begin
        ($local_controlsym)[1:N] = defaults, [bounds = bounds, localcontrol = true]
        ($timepoint_sym)[1:N] = timepoints, [shooting = true, tunable = false]
    end
    append!(new_parameters, ps[1:2])
    defs = copy(ModelingToolkit.defaults(sys))
    delete!(defs, control_var)
    if !differential
#        push!(init_equations, control_var ~ _expand_ifelse(iv, ps[2], ps[1]))
        append!(
            new_equations, [
                D(control_var) ~ 0,
            ],
        )
    else
        @warn "Differential formulation for ShootingControl variable $(control_var) is not implemented. Skipping."
    end
    controlsys = ODESystem(
        new_equations,
        ModelingToolkit.independent_variable(sys), [], new_parameters;
        name=nameof(sys),
        defaults = defs,
        initialization_eqs=init_equations,
        discrete_events=callback_eqs,
    )
    extend(sys, controlsys)
end
