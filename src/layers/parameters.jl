"""
$(TYPEDEF)

A layer to store tunable parameters.

# Fields
$(FIELDS)

It is callable with the signature `index, ps, st`.
"""
@concrete terse struct ParameterCollection <: LuxCore.AbstractLuxLayer
    "Initializer for the parameters `params = init_parameters(rng, out_dims)`"
    init_parameters
    "Dimension of the parameters"
    out_dims
end

LuxCore.initialparameters(rng::Random.AbstractRNG, x::ParameterCollection) = (; ps = x.init_parameters(rng, x.out_dims))
LuxCore.initialstates(::Random.AbstractRNG, ::ParameterCollection) = (;)

function (::ParameterCollection)(index, ps, st::NamedTuple)
    return getindex(ps.ps, index), st
end

"""
$(TYPEDEF)

A layer which moves tunable parameters into the state.

# Fields
$(FIELDS)
"""
@concrete terse struct FixedLayer <: LuxCore.AbstractLuxWrapperLayer{:layer}
    "Internal layer"
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
