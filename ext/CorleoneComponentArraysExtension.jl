module CorleoneComponentArraysExtension

using Corleone
using ComponentArrays

# ComponentVectors encode parameter fields in their axes. Traverse those fields
# through views; materializing NamedTuple(ps) would copy numeric parameter arrays.
function Corleone.parameter_bound_tree(ps::ComponentVector, value)
    names = propertynames(ps)
    isempty(names) && !isempty(ps) && return fill(value, size(ps))
    return NamedTuple{names}(
        map(names) do name
            Corleone.parameter_bound_tree(getproperty(ps, name), value)
        end
    )
end

Corleone.empty_parameter_structure(ps::ComponentVector) =
    isempty(propertynames(ps)) && isempty(ps)

Corleone.unbounded_parameter_bounds(ps::ComponentVector, value) =
    Corleone.parameter_bound_tree(ps, value)

# Splitting intervals changes axes, so rebuild structured parameter storage.
function Corleone.replace_parameter_fields(ps::ComponentVector, fields::NamedTuple)
    names = propertynames(ps)
    tree = NamedTuple{names}(map(name -> getproperty(ps, name), names))
    return ComponentArray(merge(tree, fields))
end

end
