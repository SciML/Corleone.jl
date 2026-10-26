"""
$(SIGNATURES)

Return lower optimization bounds with the parameter layout and array shapes of
`ps`. Containers and wrappers recurse using Lux's declared child fields; NamedTuple,
Tuple and array collections recurse over their layers. Generic leaves have unbounded
parameters (`-Inf`). Empty parameter subtrees remain empty.

`ShootingLayer` specializes this operation using its `ControlSymbolCache`, repeating
each symbol's lower bound across its scheduled values. Other layers may specialize
this function to supply their own bounds. Initialized parameters are supplied
explicitly; no setup, RNG draws, initializers, or solves are performed. `st` is
optional and is forwarded to child layers when supplied.
"""
function get_lower_bound end

"""
$(SIGNATURES)

Return upper optimization bounds, following the same recursive layer and parameter
structure as [`get_lower_bound`](@ref). Generic leaves use `Inf`; `ShootingLayer`
uses the upper bounds in its symbolic cache. No parameter initialization or solve
is performed.
"""
function get_upper_bound end

"""
$(SIGNATURES)

Return `(get_lower_bound(layer, ps, st), get_upper_bound(layer, ps, st))`. The
returned trees follow the supplied parameter ordering and shapes; fixed data in
Lux state is excluded. The two directional functions recursively traverse Lux
containers, wrappers, and layer collections. Only `ShootingLayer` has a concrete
layer specialization in Corleone, using its symbolic bounds metadata.

Pass initialized parameters explicitly. Native NamedTuple parameters and structured
arrays such as ComponentArrays are supported; flatten all three trees with the
same vectorization outside this API. `st` is optional. Flat optimizer vectors must
be reconstructed before requesting bounds for a container layer.
"""
function get_bounds(layer::Union{LuxCore.AbstractLuxLayer, NamedTuple, Tuple, AbstractArray, Nothing},
        ps::Union{NamedTuple, Tuple, AbstractArray}, st = nothing)
    return get_lower_bound(layer, ps, st), get_upper_bound(layer, ps, st)
end

# Mirror LuxCore.initialparameters: containers wrap declared fields, wrappers
# delegate without an extra field, and NamedTuple/Tuple collections retain shape.
for (operation, unbounded) in ((:get_lower_bound, -Inf), (:get_upper_bound, Inf))
    @eval begin
        $(operation)(::Union{LuxCore.AbstractLuxLayer, Nothing}, ps, st = nothing) =
            unbounded_parameter_bounds(ps, $unbounded)

        function $(operation)(layer::LuxCore.AbstractLuxContainerLayer{fields}, ps,
                st = nothing) where {fields}
            return container_parameter_bounds($(operation), layer, ps, st, fields)
        end

        function $(operation)(layer::LuxCore.AbstractLuxWrapperLayer{field}, ps,
                st = nothing) where {field}
            # Wrappers may move all inner parameters into state (e.g. FixedLayer).
            # An empty supplied subtree has no optimization entries to traverse.
            empty_parameter_structure(ps) && return (;)
            return $(operation)(getfield(layer, field), ps, st)
        end

        function $(operation)(layers::NamedTuple, ps, st = nothing)
            names = optimization_parameter_names(ps, keys(layers))
            return NamedTuple{names}(map(names) do name
                $(operation)(layers[name], getproperty(ps, name), bound_child_state(st, name))
            end)
        end

        function $(operation)(layers::Tuple, ps::Tuple, st = nothing)
            length(layers) == length(ps) || throw(ArgumentError("Layer and parameter tuple lengths must match."))
            return ntuple(length(layers)) do i
                $(operation)(layers[i], ps[i], bound_child_state(st, i))
            end
        end

        function $(operation)(layers::AbstractArray, ps::AbstractArray, st = nothing)
            axes(layers) == axes(ps) || throw(ArgumentError("Layer and parameter collection axes must match."))
            return map(CartesianIndices(layers)) do i
                $(operation)(layers[i], ps[i], bound_child_state(st, i))
            end
        end
    end
end

bound_child_state(st, name::Symbol) = isnothing(st) ? nothing : getproperty(st, name)
bound_child_state(st, i::Union{Integer, CartesianIndex}) = isnothing(st) ? nothing : st[i]

function container_parameter_bounds(operation, layer, ps, st, fields)
    names = optimization_parameter_names(ps, fields)
    return NamedTuple{names}(map(names) do name
        operation(getfield(layer, name), getproperty(ps, name), bound_child_state(st, name))
    end)
end

empty_parameter_structure(ps) = isempty(ps)

# Empty parameter views can represent parameterless Lux leaves. Structured array
# representations may override this hook to retain named, zero-length fields.
unbounded_parameter_bounds(ps, value) = parameter_bound_tree(ps, value)
unbounded_parameter_bounds(ps::AbstractArray, value) =
    isempty(ps) ? (;) : parameter_bound_tree(ps, value)

parameter_bound_tree(ps::NamedTuple, value) = map(x -> parameter_bound_tree(x, value), ps)
parameter_bound_tree(ps::Tuple, value) = map(x -> parameter_bound_tree(x, value), ps)
parameter_bound_tree(ps::AbstractArray{<:Number}, value) = fill(value, size(ps))
parameter_bound_tree(ps::AbstractArray, value) = map(x -> parameter_bound_tree(x, value), ps)
parameter_bound_tree(ps::Number, value) = value

function optimization_parameter_names(ps, expected)
    isempty(expected) && isempty(ps) && return ()
    names = propertynames(ps)
    length(names) == length(expected) && all(name -> name in expected, names) ||
        throw(ArgumentError("Expected structured Lux parameter fields $expected, got $names. Reconstruct flat optimizer vectors before requesting bounds."))
    return names
end

# Concrete implementations live beside their layer or trajectory type.
function get_number_of_shooting_constraints end
function shooting_constraints end
function shooting_constraints! end
