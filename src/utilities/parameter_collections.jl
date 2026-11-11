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
function parameter_collections(layer::LuxCore.AbstractLuxContainerLayer{fields}, ps) where {fields}
    names = optimization_parameter_names(ps, fields)
    return parameter_collections(layer, ps, Val(names))
end
function parameter_collections(layers::NamedTuple, ps)
    names = optimization_parameter_names(ps, keys(layers))
    return parameter_collections(layers, ps, Val(names))
end

# Literal field accesses and tuple splats preserve heterogeneous collection types.
@generated function parameter_collections(layers, ps, ::Val{names}) where {names}
    entries = [Expr(:..., :(parameter_collections(getfield(layers, $(QuoteNode(name))), getproperty(ps, $(QuoteNode(name)))))) for name in names]
    return Expr(:tuple, entries...)
end

function collection_log_prior(collection, values)
    isnothing(collection.prior) && return 0.0
    return sum(values.ps; init = 0.0) do x
        log_density(collection.prior, natural_value(collection.transform, x)) +
            log_jacobian(collection.transform, x)
    end
end

prior_expression(::Type, layer, ps) = :(0.0)
prior_expression(::Type{<:FixedLayer}, layer, ps) = :(0.0)
prior_expression(::Type{<:ParameterCollection}, layer, ps) = :(collection_log_prior($layer, $ps))
prior_expression(::Type{L}, layer, ps) where {L <: NamedTuple} = container_prior_expression(L, fieldnames(L), layer, ps)
function prior_expression(::Type{L}, layer, ps) where {field, L <: LuxCore.AbstractLuxWrapperLayer{field}}
    child = :(getfield($layer, $(QuoteNode(field))))
    child_type = fieldtype(L, field)
    return isconcretetype(child_type) ? prior_expression(child_type, child, ps) : :(log_prior($child, $ps))
end
function prior_expression(::Type{L}, layer, ps) where {fields, L <: LuxCore.AbstractLuxContainerLayer{fields}}
    return container_prior_expression(L, fields, layer, ps)
end
function container_prior_expression(::Type{L}, fields, layer, ps) where {L}
    terms = map(fields) do field
        child = :(getfield($layer, $(QuoteNode(field))))
        parameters = :(getproperty($ps, $(QuoteNode(field))))
        child_type = fieldtype(L, field)
        isconcretetype(child_type) ? prior_expression(child_type, child, parameters) : :(log_prior($child, $parameters))
    end
    return quote
        optimization_parameter_names($ps, $(QuoteNode(fields)))
        sum(($(terms...),); init = 0.0)
    end
end

# Flatten the density traversal into literal field accesses. Recursive tuple
# collection widens deeply nested ComponentArray view types during inference.
@generated function log_prior(layer::L, ps) where {L}
    return prior_expression(L, :layer, :ps)
end
