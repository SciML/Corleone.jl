"""
$(TYPEDEF)

A layer storing tunable values initialized during Lux setup.

# Fields
$(FIELDS)

Calling with `(index, ps, st)` returns `(ps.ps[index], st)`.
"""
@concrete terse struct ParameterCollection <: LuxCore.AbstractLuxLayer
    "The initializer called as `init_parameters(rng, out_dims)` during Lux setup."
    init_parameters
    "The number of stored parameter values."
    out_dims
end

LuxCore.initialparameters(rng::Random.AbstractRNG, x::ParameterCollection) = (; ps = x.init_parameters(rng, x.out_dims))
LuxCore.initialstates(::Random.AbstractRNG, ::ParameterCollection) = (;)
# Shape queries must not execute user initializers or consume RNG values.
LuxCore.parameterlength(x::ParameterCollection) = x.out_dims
LuxCore.statelength(::ParameterCollection) = 0

function (::ParameterCollection)(index, ps, st::NamedTuple)
    return getindex(ps.ps, index), st
end

"""
$(TYPEDEF)

A wrapper that stores the inner layer's parameters and state in Lux state.
The wrapper has no trainable parameters. Calling it forwards its stored values
to the inner layer and retains that layer's updated state.

# Fields
$(FIELDS)
"""
@concrete terse struct FixedLayer <: LuxCore.AbstractLuxWrapperLayer{:layer}
    "The inner layer whose parameters are held fixed in Lux state."
    layer
end


LuxCore.initialparameters(::Random.AbstractRNG, ::FixedLayer) = (;)
LuxCore.parameterlength(::FixedLayer) = 0
LuxCore.statelength(x::FixedLayer) = LuxCore.parameterlength(x.layer) + LuxCore.statelength(x.layer)
LuxCore.initialstates(rng::Random.AbstractRNG, x::FixedLayer) = (;
    states = LuxCore.initialstates(rng, x.layer),
    parameters = LuxCore.initialparameters(rng, x.layer),
)

function (x::FixedLayer)(val, ps, st::NamedTuple)
    out, new_st = x.layer(val, st.parameters, st.states)
    return out, (; states = new_st, parameters = st.parameters)
end
