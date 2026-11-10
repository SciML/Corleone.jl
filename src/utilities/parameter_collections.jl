collection_transform(x::ParameterCollection) = x.transform
collection_transform(x::Union{InitialCondition, PiecewiseConstant}) = collection_transform(x.parameter)
collection_transform(::Any) = BoundedTransform()

"""
$(SIGNATURES)

Return `(collection, ps)` pairs for the tunable parameter collections of `layer`, in
the order of the Lux parameters `ps`. Fixed collections are skipped.
"""
parameter_collections(::Any, ps) = ()
parameter_collections(layer::ParameterCollection, ps) = ((layer, ps),)
parameter_collections(::FixedLayer, ps) = ()
parameter_collections(layer::LuxCore.AbstractLuxWrapperLayer{field}, ps) where {field} =
    parameter_collections(getfield(layer, field), ps)
parameter_collections(layer::LuxCore.AbstractLuxContainerLayer{fields}, ps) where {fields} =
    child_collections(map(name -> getfield(layer, name), fields), ps, fields)
parameter_collections(layers::NamedTuple, ps) = child_collections(values(layers), ps, keys(layers))

function child_collections(layers, ps, names)
    children = map((layer, name) -> parameter_collections(layer, getproperty(ps, name)), layers, names)
    return reduce((a, b) -> (a..., b...), children; init = ())
end
