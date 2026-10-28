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

# Resolve the lookup once, then sample scalar buffers using the same index.
apply_scalar_collections(layer::LuxCore.AbstractLuxLayer, index, ps, st) = LuxCore.apply(layer, index, ps, st)

@generated function apply_scalar_collections(layers::NamedTuple{names}, index, ps, st) where {names}
    outs = [gensym(:out) for _ in names]
    states = [gensym(:state) for _ in names]
    calls = [:(($(outs[i]), $(states[i])) = layers.$name(index, ps.$name, st.$name)) for (i, name) in enumerate(names)]
    return quote
        $(calls...)
        return [$(outs...)], NamedTuple{$names}(($(states...),))
    end
end

# Construction metadata. Every numerical leaf still owns a scalar-valued vector.
function group_scalar_collections(entries)
    groups = Vector{eltype(entries)}[]
    for entry in entries
        position = isnothing(entry.parameter) ? nothing : findfirst(groups) do group
                !isnothing(first(group).parameter) && isequal(first(group).grid, entry.grid)
        end
        isnothing(position) ? push!(groups, [entry]) : push!(groups[position], entry)
    end
    return groups
end

collection_permutation(indices) = sortperm(reduce(vcat, indices; init = Int[]))
number_of_tunable_collections(::ParameterCollection) = 1
number_of_tunable_collections(::FixedLayer) = 0
number_of_tunable_collections(layers::NamedTuple) = sum(number_of_tunable_collections, values(layers); init = 0)

function collection_scalar_type(layer::ParameterCollection, ps, st)
    values = ps.ps
    isempty(values) && return eltype(values)
    T = eltype(values)
    return isconcretetype(T) ? typeof(first(values)) : mapreduce(typeof, promote_type, values)
end
collection_scalar_type(layer::FixedLayer, ps, st) = collection_scalar_type(layer.layer, st.parameters, st.states)
@generated function collection_scalar_type(layers::NamedTuple{names}, ps, st) where {names}
    types = [:(collection_scalar_type(layers.$name, ps.$name, st.$name)) for name in names]
    return :(promote_type($(types...)))
end
ChainRulesCore.@non_differentiable collection_scalar_type(layer, ps, st)

# Called only while generating code: scalar extraction precedes vector assembly.
function child_scalar_expressions(type, out)
    type <: FixedLayer && return child_scalar_expressions(fieldtype(type, :layer), out)
    :parameter in fieldnames(type) || return [out]
    collection = fieldtype(type, :parameter)
    return collection <: NamedTuple ? [:($out[$i]) for i in 1:fieldcount(collection)] : [out]
end

function canonical_output_expression(types, outs)
    scalars = reduce(vcat, map(child_scalar_expressions, types, outs); init = Any[])
    flat = gensym(:flat)
    promoted = gensym(:promoted)
    assembly = Expr(:vect, [:(convert($promoted, $flat[permutation[$i]])) for i in eachindex(scalars)]...)
    return Expr(:block, :($flat = ($(scalars...),)), :($promoted = Base.promote_typeof($flat...)), assembly)
end
