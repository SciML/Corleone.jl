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

# Materialize the integer index tree once instead of rebuilding nested views at
# every shooting boundary. Numeric parameters retain their existing view path.
Corleone.sparsity_indices(ids::ComponentVector) = NamedTuple(ids)

# Splitting intervals changes axes, so rebuild structured parameter storage.
function Corleone.replace_parameter_fields(ps::ComponentVector, fields::NamedTuple)
    names = propertynames(ps)
    tree = NamedTuple{names}(map(name -> getproperty(ps, name), names))
    return ComponentArray(merge(tree, fields))
end

end
