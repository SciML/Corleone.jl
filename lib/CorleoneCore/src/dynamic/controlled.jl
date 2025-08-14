"""
$(TYPEDEF)

# Fields
$(FIELDS)
"""
struct ControlledDynamics{F, L} <: LuxCore.AbstractLuxContainerLayer{(:dynamics, :controls,)}
    "The dynamic functions"
    dynamics::F 
    "The control approximations"
    controls::L 
end    

LuxCore.initialparameters(rng::Random.AbstractRNG, x::ControlledDynamics{<:Function}) = (; dynamics = (;), controls = LuxCore.initialparameters(rng, x.controls)) 
LuxCore.initialstates(rng::Random.AbstractRNG, x::ControlledDynamics{<:Function}) = (; dynamics = (;), controls = LuxCore.initialstates(rng, x.controls))

SciMLBase.isinplace(model::ControlledDynamics{<:Function}, nparams, args...; kwargs...) = SciMLBase.isinplace(model.dynamics, nparams+1, args...; kwargs...)

function (model::ControlledDynamics{<: Function})(args::Tuple, ps, st)
    (; dynamics, controls) = model
    u, control_st = controls(args, ps.controls, st.controls)
    out = dynamics(args..., ps.dynamics, u)
    return out, (; dynamics = st.dynamics, controls = control_st)
end

function (model::ControlledDynamics)(args::Tuple, ps, st)
    (; dynamics, controls) = model
    u, control_st = controls(args, ps.controls, st.controls)
    out, dynamics_st = dynamics((args..., u), ps.dynamics, st.dynamics)
    return out, (; dynamics = dynamics_st, controls = control_st)
end