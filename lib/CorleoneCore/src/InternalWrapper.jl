module InternalWrapper

using SciMLBase
using DocStringExtensions
using LuxCore

is_extension_loaded(::Val) = false

"""
$(FUNCTIONNAME)

Defines the initialization of the underlying model. Needs a custom dispatch for all models.

As a default it supports any `DEFunction` (which returns just the function itsself.)
"""
initialize_model(model, args...) = throw(ArgumentError("The model $(model) has no initialization routine defined!"))


# Right now, we simply wrap this in a StatefulLayer like struct 
# This assumes that all states are constant!
struct WrappedModel{M,S}
    model::M
    states::S
end

function wrap_stateful end

(m::WrappedModel)(args...) = begin 
    states, p, t = args
    return first(m.model((states..., t), p, m.st))
end

function initialize_model(d::LuxCore.AbstractLuxLayer, ps, st::NamedTuple, args...)
    is_extension_loaded(Val{:Lux}()) && return wrap_stateful(d, ps, st, args...)
    @warn "Using internal wrapper to wrap function. Please load Lux.jl for using the StatefulLuxLayer." maxlog = 1
    WrappedModel(d, st)
end

export initialize_model

end
