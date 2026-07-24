@concrete terse struct TimedFunction <: LuxCore.AbstractLuxWrapperLayer{(:layer)}
    "The layer"
    layer
    "The timepoints used in the layer"
    timepoints
end

LuxCore.initialparameters(::Random.AbstractRNG, ::TimedFunction{<:Function}) = (;)
LuxCore.parameterlength(::Random.AbstractRNG, ::TimedFunction{<:Function}) = 0
LuxCore.initialstates(::Random.AbstractRNG, ::TimedFunction{<:Function}) = (;)
LuxCore.statelength(::Random.AbstractRNG, ::TimedFunction{<:Function}) = 0

function get_timepoints(layer::TimedFunction{T}, ps, st) where {T}
    return if isa(T, LuxCore.AbstractLuxLayer)
        vcat(get_timepoints(layer.layer, ps, st), layer.timepoints)
    else
        layer.timepoints
    end
end

(layer::TimedFunction{<:Function})(traj, ps, st) = layer.layer(traj), st
(layer::TimedFunction{<:LuxCore.AbstractLuxLayer})(traj, ps, st) = layer.layer(traj, ps, st)

@concrete terse struct TimedFunctions <: LuxCore.AbstractLuxContainerLayer{(:layers)}
    "The functions"
    layers
end

function (layer::TimedFunctions)(traj, ps, st)
    return apply_timedfunctions(layer.layers, traj, ps, st)
end
