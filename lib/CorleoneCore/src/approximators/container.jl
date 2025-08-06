"""
$(TYPEDEF)

A container for multiple signals
"""
struct SignalContainer{N,A,L} <: LuxCore.AbstractLuxContainerLayer{(:layers,)}
    "The name of the container"
    name::N
    "The aggregation for all parameters. Defaults to `identity`, which returns a named tuple"
    aggregation::A
    "The individual layers"
    layers::L
end

function SignalContainer(layers::NamedTuple; name=nothing, aggregation::Function=identity, kwargs...)
    return SignalContainer{typeof(name),typeof(aggregation),typeof(layers)}(name, aggregation, layers)
end

function SignalContainer(layers::LuxCore.AbstractLuxLayer...; kwargs...)
    layers = NamedTuple{ntuple(i -> Symbol(:layer_, i), length(layers))}(layers)
    SignalContainer(layers; kwargs...)
end

function (container::SignalContainer)(args::Tuple, ps, st::NamedTuple)
    output, layer_st = _apply_parameters(container.layers, args, ps.layers, st.layers)
    out = container.aggregation(output)
    return out, NamedTuple{(:layers,)}((layer_st,))
end

@generated function _apply_parameters(layers::NamedTuple{fields}, t, ps, st::NamedTuple{fields}) where {fields}
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

