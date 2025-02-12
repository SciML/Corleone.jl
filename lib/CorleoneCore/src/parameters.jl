"""
$(TYPEDEF)

A simple container for [`Parameter`](@ref) layers. This layer **must** be called with `t`.

In addition, the parameter
"""
struct ParameterContainer{N, P <: NamedTuple, O} <: LuxCore.AbstractLuxContainerLayer{(:layers,)}
    "The name of the parameter container"
    name::N
    "The individual parameter layers"
    layers::P
    "The output function, which defines how the individual layers should be combined. Defaults to `identity` which returns a `NamedTuple`."
    output::O
end

function ParameterContainer(layers::Parameter...; name = nothing, output = identity, kwargs...)
    parameters = NamedTuple{ntuple(i->Symbol(:layer_, i), length(layers))}(layers)
    return ParameterContainer{typeof(name), typeof(parameters), typeof(output)}(name, parameters, output)
end

function (container::ParameterContainer)(t, ps, st::NamedTuple)
    output, layer_st =  _apply_parameters(container.layers, t, ps.layers, st.layers)
    out = container.output(output)
    return out, NamedTuple{(:layers,)}((layer_st,))
end

@generated function _apply_parameters(layers::NamedTuple{fields}, t, ps, st::NamedTuple{fields}) where fields 
    N = length(fields)
    outs = [gensym() for _ in Base.OneTo(N)]
    sts = [gensym() for _ in Base.OneTo(N)]
    exprs = Expr[]
    for i in 1:N
        push!(exprs, 
            :(($(outs[i]), $(sts[i])) = layers.$(fields[i])(t, ps.$(fields[i]), st.$(fields[i])))
        )
    end
    push!(exprs, :(st_ = NamedTuple{fields}(($(sts...),))))
    push!(exprs, :(out_ = NamedTuple{fields}(($(outs...),))))
    push!(exprs, :(return (out_, st_)))
    return Expr(:block, exprs...)
end
