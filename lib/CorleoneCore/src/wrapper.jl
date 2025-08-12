"""
$(FUNCTIONNAME)

Defines the initialization of the underlying model. Needs a custom dispatch for all models
As a default it supports any `DEFunction` (which returns just the function itsself.)
"""
wrap_model(model, args...) = throw(ArgumentError("The model $(model) has no initialization routine defined!"))

"""
$(TYPEDEF)

Similar to the Lux.StatefulLuxLayer, but less complete in its implementation. 

# Fields 
$(FIELDS)
"""
mutable struct StatefulWrapper{M, S}
    "The underlying model"
    const model::M 
    "The current state if applicable. Is `nothing` if no state is needed."
    state::S
end 

(x::StatefulWrapper{<:Any, Nothing})(args...) = x.model(args...)

SciMLBase.isinplace(model::StatefulWrapper, args...; kwargs...) = SciMLBase.isinplace(model.model, args...; kwargs...)

function (s::StatefulWrapper)(x, p)
    y, st = LuxCore.apply(s.model, x, p, s.state)
    s.state = st 
    return y 
end

function (model::StatefulWrapper)(x::AbstractArray, ps, t)
    model((x,t), ps)
end

function (model::StatefulWrapper)(dx, x::AbstractArray, ps, t)
    model((dx, x,t), ps)
end

function (model::StatefulWrapper)(res, dx, x::AbstractArray, ps, t)
    model((res, dx, x,t), ps)
end


wrap_model(f::SciMLBase.AbstractDiffEqFunction, args...) = StatefulWrapper(f, nothing)

# This is overwritten / dispatch on
function stateful_model end

function wrap_model(d::LuxCore.AbstractLuxLayer, ps, st::NamedTuple, args...)
    if is_extension_loaded(Val{:Lux}()) 
        model = stateful_model(d, ps, st, args...)
        return StatefulWrapper(model, nothing)
    end
    @warn "Using internal wrapper to wrap function. Please load Lux.jl for using the StatefulLuxLayer." maxlog = 1
    StatefulWrapper(d, st)
end

