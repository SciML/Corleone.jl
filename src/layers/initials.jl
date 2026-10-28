"""
$(TYPEDEF)

A single-variable layer that returns the incoming value and Lux state unchanged.
It has no trainable values or scheduled reset times.

# Fields
$(FIELDS)
"""
@concrete terse struct PassThrough <: LuxCore.AbstractLuxLayer
    "The optional name of the state variable."
    name
end

(layer::PassThrough)((incoming, t)::Tuple, ps, st) = incoming, st
get_timestops(::PassThrough, ps, st, ::Type{T}) where {T} = T[]

"""
$(TYPEDEF)

Values for a single state variable at the specified times. There is one value per
entry in `timegrid`, initialized from a scalar, vector, or callback accepting
`(rng, n::Int)` or `(rng, times::AbstractVector)`. Count callbacks take precedence;
time-based callbacks receive the grid in its supplied order.
Calling the layer with `(incoming_value, t)` returns the value at the matching grid
index, or `incoming_value` when no time matches. A `nothing` grid is pass-through
and allocates no trainable values.

# Fields
$(FIELDS)
"""
@concrete terse struct InitialCondition <: LuxCore.AbstractLuxContainerLayer{(:lookup, :parameter)}
    "The optional name of the state variable."
    name
    "Exact-time lookup mapping each scheduled reset time to its value index."
    lookup
    "A scalar parameter collection or NamedTuple of scalar collections sharing this reset grid."
    parameter
end

"""
$(SIGNATURES)

Construct reset values for one state variable on `timegrid`. `default` is a scalar,
vector, or count/timepoint initializer as described by [`Tunable`](@ref). Times must
be unique; `nothing` selects an empty reset grid. `cache` is an optional lookup
cache. Output length and `bounds` are checked during Lux setup.
"""
function InitialCondition(
        timegrid::AbstractVector{<:Real}, default;
        name = nothing, cache = nothing, bounds = (-Inf, Inf)
    )
    allunique(timegrid) || throw(ArgumentError("Initial-condition times must be unique."))
    lookup = LookUp(Dict(t => i for (i, t) in enumerate(timegrid)), cache)
    initialize = (rng, n) -> initialize_annotation_values(default, rng, n, timegrid, bounds)
    return InitialCondition(name, lookup, ParameterCollection(initialize, length(timegrid)))
end

function InitialCondition(::Nothing, default; kwargs...)
    return InitialCondition(Float64[], default; kwargs...)
end


function InitialCondition(timegrid::AbstractVector{<:Real}, parameters::NamedTuple; name = nothing, cache = nothing)
    grid = isnothing(timegrid) ? Float64[] : timegrid
    allunique(grid) || throw(ArgumentError("Initial-condition times must be unique."))
    return InitialCondition(name, LookUp(Dict(t => i for (i, t) in enumerate(grid)), cache), parameters)
end

function LuxCore.initialstates(rng::Random.AbstractRNG, x::InitialCondition)
    return (;
        lookup = LuxCore.initialstates(rng, x.lookup),
        parameter = LuxCore.initialstates(rng, x.parameter),
        active = false,
    )

end

function (layer::InitialCondition)((incoming, t)::Tuple, ps, st)
    T = promote_type(typeof(incoming), eltype(ps.parameter.ps))
    haskey(layer.lookup, t, ps.lookup, st.lookup) || return T(incoming), merge(st, (; active = false))
    (index, _), lookup_st = LuxCore.apply(layer.lookup, t, ps.lookup, st.lookup)
    value, parameter_st = LuxCore.apply(layer.parameter, index, ps.parameter, st.parameter)
    # Both reset and pass-through values must retain the same promoted scalar type.
    return T(value), (; lookup = lookup_st, parameter = parameter_st, active = true)
end

function (layer::InitialCondition)((incoming, t)::Tuple{<:AbstractVector, Any}, ps, st)
    T = promote_type(eltype(incoming), collection_scalar_type(layer.parameter, ps.parameter, st.parameter))
    haskey(layer.lookup, t, ps.lookup, st.lookup) || return [T(value) for value in incoming], merge(st, (; active = false))
    (index, _), lookup_st = LuxCore.apply(layer.lookup, t, ps.lookup, st.lookup)
    out, parameter_st = apply_scalar_collections(layer.parameter, index, ps.parameter, st.parameter)
    return promote_reset_values(T, out), (; lookup = lookup_st, parameter = parameter_st, active = true)
end
promote_reset_values(::Type{T}, values::Vector{T}) where {T} = values
promote_reset_values(::Type{T}, values) where {T} = [T(value) for value in values]

istypemax(x::V) where {V} = x === typemax(V)

get_timestops(layer::InitialCondition, ps, st, ::Type{T}) where {T} =
    get_timestops(layer.lookup, nothing, st.lookup, T)

"""
$(TYPEDEF)

A collection of shared-grid `InitialCondition` layers, analogous to `Parameter`.
Definitions are `annotation => timegrid` pairs; plain indices and symbols are also
accepted as tunable annotations. Variables with no grid pass through their incoming
values. `Constant` annotations store scheduled values in Lux state via `FixedLayer`.

Calling with `(incoming_state, t)` returns `((state, residuals), next_st)`. Residuals
are assigned values minus the corresponding incoming values, in state index
order. Only active, non-fixed layers contribute residuals.
Set `matching = false` to apply assignments without producing matching residuals,
as used for free initial values at the global problem start.
The Boolean vector `st.active` is allocated during setup and reused on each call.
Updating this scheduling metadata is non-differentiable; residual arithmetic is not.

# Fields
$(FIELDS)
"""
@concrete terse struct InitialConditions <: LuxCore.AbstractLuxContainerLayer{(:initialconditions,)}
    "The optional name of the initial-condition collection."
    name
    "Reconstructs the problem state from its canonical tunable values."
    repack
    "Shared-grid reset layers and pass-through layers, keyed by their first variable symbol."
    initialconditions
    "Original state indices represented by each reset or pass-through layer."
    indices
end

InitialConditions(name, repack, initialconditions) =
    InitialConditions(name, repack, initialconditions, Tuple([i] for i in 1:length(initialconditions)))

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::InitialConditions)
    return (;
        initialconditions = map(ic -> LuxCore.initialstates(rng, ic), layer.initialconditions),
        active = fill(false, sum(length, layer.indices; init = 0)),
        permutation = collection_permutation(layer.indices),
    )
end

"""
$(SIGNATURES)

Construct per-variable initial conditions from `annotation => timegrid` definitions.
Annotations of kind `Val(:variable)` select state entries. Unspecified variables
pass through their incoming values; `Constant` definitions store reset values in
Lux state. `sys` supplies symbols, defaults, and annotation bounds.
"""
function InitialConditions(
        problem::SciMLBase.AbstractSciMLProblem, definitions...;
        name = nothing, sys = get_symbolic_container(problem)
    )
    vals, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), problem.u0)
    processed = reduce(
        vcat, map(definitions) do definition
            process_definition(sys, Val(:variable), definition...)
        end; init = Pair[]
    )
    indices = map(definition -> get_index(first(definition)), processed)
    allunique(indices) || throw(ArgumentError("Define each initial-condition variable only once."))
    symbols = SymbolicIndexingInterface.variable_symbols(sys)
    entries = map(eachindex(vals)) do i
        position = findfirst(==(i), indices)
        sym = only(filter(s -> SymbolicIndexingInterface.variable_index(sys, s) == i, symbols))
        if isnothing(position)
            return (; index = i, name = Symbol(sym), grid = nothing, parameter = nothing, fixed = false)
        end
        annotation, grid = processed[position]
        grid = isnothing(grid) ? Float64[] : grid
        allunique(grid) || throw(ArgumentError("Initial-condition times must be unique."))
        default, bounds = get_default(annotation), get_bounds(annotation)
        initialize = (rng, n) -> initialize_annotation_values(default, rng, n, grid, bounds)
        return (;
            index = i, name = Symbol(sym), grid,
            parameter = ParameterCollection(initialize, length(grid)), fixed = !is_tunable(annotation),
        )
    end
    groups = group_scalar_collections(entries)
    conditions = map(groups) do group
        entry = first(group)
        if isnothing(entry.parameter)
            entry.name => PassThrough(entry.name)
        elseif length(group) == 1
            ic = InitialCondition(entry.name, LookUp(Dict(t => i for (i, t) in enumerate(entry.grid)), nothing), entry.parameter)
            entry.name => (entry.fixed ? FixedLayer(ic) : ic)
        else
            children = NamedTuple(Tuple(item.name => (item.fixed ? FixedLayer(item.parameter) : item.parameter) for item in group))
            entry.name => InitialCondition(entry.grid, children; name = entry.name)
        end
    end |> NamedTuple
    return InitialConditions(name, repack, conditions, Tuple([entry.index for entry in group] for group in groups))
end

function get_timestops(layer::InitialConditions, ps, st, ::Type{T}) where {T}
    times = reduce(
        vcat, map(keys(layer.initialconditions)) do key
            get_timestops(layer.initialconditions[key], nothing, st.initialconditions[key], T)
        end; init = T[]
    )
    return sort!(unique!(times))
end


is_active(::InitialCondition, st) = st.active


initial_condition_activity(::PassThrough, st) = (false,)
initial_condition_activity(::FixedLayer, st) = (false,)
function initial_condition_activity(ic::InitialCondition, st)
    return ic.parameter isa NamedTuple ? map(value -> st.active && !(value isa FixedLayer), values(ic.parameter)) : (st.active,)
end
"""
$(SIGNATURES)

Update the preallocated residual activity mask from per-variable layers and their
new Lux states. Only active, non-fixed initial conditions contribute residuals.
This scheduling update is non-differentiable; an all-`PassThrough` tuple clears
the mask without checking individual layers.
"""
function update_active_ics!(active, initial_conditions::Tuple, new_st::Tuple, permutation = Base.OneTo(length(active)))
    parts = map(initial_condition_activity, initial_conditions, new_st)
    flags = foldl((a, b) -> (a..., b...), parts; init = ())
    foreach(eachindex(active)) do i
        active[i] = flags[permutation[i]]
    end
    return active
end
update_active_ics!(active, ::Tuple{Vararg{PassThrough}}, ::Tuple, permutation = nothing) = active .= false
ChainRulesCore.@non_differentiable update_active_ics!(active, initial_conditions, new_st)
ChainRulesCore.@non_differentiable update_active_ics!(active, initial_conditions, new_st, permutation)

function (layer::InitialConditions)((incoming, t)::Tuple, ps, st; matching::Bool = true)
    out, new_st = apply_ics(layer.initialconditions, incoming, t, ps.initialconditions, st.initialconditions, layer.indices, st.permutation)
    active = update_active_ics!(st.active, values(layer.initialconditions), values(new_st), st.permutation)
    residuals = [out[i] - incoming[i] for i in eachindex(out) if matching && active[i]]
    return (layer.repack(out), residuals), merge(st, (; initialconditions = new_st, active))
end

@generated function apply_ics(ics::NamedTuple{names}, vals, t, ps, st::NamedTuple{names}, indices, permutation) where {names}
    outs = Symbol[]
    sts = Symbol[]
    exprs = Expr[]
    for (i, n) in enumerate(names)
        push!(outs, gensym(:out))
        push!(sts, gensym(:st))
        type = fieldtype(ics, i)
        grouped = length(child_scalar_expressions(type, :out)) > 1
        incoming = grouped ? :(view(vals, indices[$i])) : :(vals[only(indices[$i])])
        push!(exprs, :(($(last(outs)), $(last(sts))) = ics.$n(($incoming, t), ps.$n, st.$n)))
    end
    push!(
        exprs,
        :(new_st = NamedTuple{$(names)}(($(sts...),)))
    )
    # Children return scalars. A vector literal preserves scalar tracking;
    # ReverseDiff's vcat overload would instead track the array as a whole.
    push!(
        exprs,
        :(vals = $(canonical_output_expression(fieldtypes(ics), outs)))
    )
    push!(exprs, :(return vals, new_st))
    return Expr(:block, exprs...)
end
