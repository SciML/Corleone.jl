module CorleoneComponentArraysExtension

import Corleone
using ComponentArrays: ComponentArray, ComponentVector, getaxes, getdata
import Corleone: pack

function pack(::Val{:ComponentArrays}, ps::Union{NamedTuple, ComponentVector})
    parameters = ComponentArray(ps)
    data = getdata(parameters)
    eltype(data) <: Real || throw(ArgumentError("Optimization parameters must have a real scalar type."))
    return data, Base.Fix2(ComponentArray, getaxes(parameters))
end

# Splitting intervals changes axes, so rebuild structured parameter storage.
function Corleone.replace_parameter_fields(ps::ComponentVector, fields::NamedTuple)
    names = propertynames(ps)
    tree = NamedTuple{names}(map(name -> getproperty(ps, name), names))
    return ComponentArray(merge(tree, fields))
end

end
