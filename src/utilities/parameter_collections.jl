collection_transform(x::ParameterCollection) = x.transform
collection_transform(x::Union{InitialCondition, PiecewiseConstant}) = collection_transform(x.parameter)
collection_transform(::Any) = BoundedTransform()

"""
$(SIGNATURES)

Return `(collection, ps, inserted)` triples for the tunable parameter collections of
`layer`, in the order of the Lux parameters `ps`. `inserted` holds the indices of
values that [`add_parameter_segments!`](@ref) duplicated from their predecessor,
read from the enclosing lookup state in `st`. Fixed collections are skipped.
"""
parameter_collections(layer, ps, st) = parameter_collections(layer, ps, st, ())
parameter_collections(::Any, ps, st, inserted) = ()
parameter_collections(layer::ParameterCollection, ps, st, inserted) = ((layer, ps, inserted),)
parameter_collections(::FixedLayer, ps, st, inserted) = ()
parameter_collections(layer::PiecewiseConstant, ps, st, inserted) =
    parameter_collections(layer.parameter, ps.parameter, st.parameter, st.lookup.inserted)
parameter_collections(layer::LuxCore.AbstractLuxWrapperLayer{field}, ps, st, inserted) where {field} =
    parameter_collections(getfield(layer, field), ps, st, inserted)
function parameter_collections(layer::LuxCore.AbstractLuxContainerLayer{fields}, ps, st, inserted) where {fields}
    names = optimization_parameter_names(ps, fields)
    return parameter_collections(layer, ps, st, inserted, Val(names))
end
function parameter_collections(layers::NamedTuple, ps, st, inserted)
    names = optimization_parameter_names(ps, keys(layers))
    return parameter_collections(layers, ps, st, inserted, Val(names))
end

# Literal field accesses and tuple splats preserve heterogeneous collection types.
@generated function parameter_collections(layers, ps, st, inserted, ::Val{names}) where {names}
    entries = map(names) do name
        field = QuoteNode(name)
        Expr(:..., :(parameter_collections(getfield(layers, $field), getproperty(ps, $field), getproperty(st, $field), inserted)))
    end
    return Expr(:tuple, entries...)
end

# Duplicated segment values are held equal to their predecessor by matching
# constraints, so each prior counts once per original value.
function collection_log_prior(collection, values, inserted)
    isnothing(collection.prior) && return 0.0
    x = values.ps
    isempty(x) && return zero(eltype(x))
    # Zygote has no rule for `sum(f, itr; init)`.
    return sum(eachindex(x)) do i
        term = log_density(collection.prior, natural_value(collection.transform, x[i])) +
            log_jacobian(collection.transform, x[i])
        i in inserted ? zero(term) : term
    end
end

prior_expression(::Type, layer, ps, st, inserted) = :(0.0)
prior_expression(::Type{<:FixedLayer}, layer, ps, st, inserted) = :(0.0)
# A collection whose prior type is `Nothing` contributes a literal zero, so the
# generated traversal never touches its parameter view.
prior_expression(::Type{L}, layer, ps, st, inserted) where {L <: ParameterCollection} =
    fieldtype(L, :prior) === Nothing ? 0.0 : :(collection_log_prior($layer, $ps, $inserted))
function prior_expression(::Type{L}, layer, ps, st, inserted) where {L <: PiecewiseConstant}
    return child_prior_expression(fieldtype(L, :parameter), :($layer.parameter), :($ps.parameter), :($st.parameter), :($st.lookup.inserted))
end
prior_expression(::Type{L}, layer, ps, st, inserted) where {L <: NamedTuple} =
    container_prior_expression(L, fieldnames(L), layer, ps, st, inserted)
function prior_expression(::Type{L}, layer, ps, st, inserted) where {field, L <: LuxCore.AbstractLuxWrapperLayer{field}}
    return child_prior_expression(fieldtype(L, field), :(getfield($layer, $(QuoteNode(field)))), ps, st, inserted)
end
function prior_expression(::Type{L}, layer, ps, st, inserted) where {fields, L <: LuxCore.AbstractLuxContainerLayer{fields}}
    return container_prior_expression(L, fields, layer, ps, st, inserted)
end
function container_prior_expression(::Type{L}, fields, layer, ps, st, inserted) where {L}
    terms = map(fields) do field
        name = QuoteNode(field)
        child_prior_expression(
            fieldtype(L, field), :(getfield($layer, $name)), :(getproperty($ps, $name)), :(getproperty($st, $name)), inserted
        )
    end
    # Literal zeros (no prior below this field) are dropped from the sum. A plain
    # `+` call keeps the reduction differentiable under Zygote, whose rule for
    # `sum(::Tuple; init)` returns `nothing`.
    terms = filter(term -> term !== 0.0, collect(terms))
    return quote
        optimization_parameter_names($ps, $(QuoteNode(fields)))
        +(0.0, $(terms...))
    end
end
child_prior_expression(T, layer, ps, st, inserted) =
    isconcretetype(T) ? prior_expression(T, layer, ps, st, inserted) : :(layer_log_prior($layer, $ps, $st, $inserted))

log_prior(layer, ps, st) = layer_log_prior(layer, ps, st, ())

# Flatten the density traversal into literal field accesses. Recursive tuple
# collection widens deeply nested ComponentArray view types during inference.
@generated function layer_log_prior(layer::L, ps, st, inserted) where {L}
    return prior_expression(L, :layer, :ps, :st, :inserted)
end
