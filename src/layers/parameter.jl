"""
$(TYPEDEF)

One or more piecewise-constant parameters sharing a lookup. Lookup times separate intervals, so a grid
of `n` switching times stores `n + 1` values. A `nothing` grid stores one constant
value. Calling with `(t, ps, st)` returns `(value, next_st)`.

# Fields
$(FIELDS)
"""
@concrete terse struct PiecewiseConstant <: LuxCore.AbstractLuxContainerLayer{(:lookup, :parameter)}
    "The optional name of the parameter."
    name
    "The time lookup selecting a piecewise-constant interval."
    lookup
    "A scalar parameter collection or NamedTuple of scalar collections sharing this grid."
    parameter
end

"""
$(SIGNATURES)

Construct interval values on `timegrid` from a scalar, vector, or count/timepoint
initializer as described by [`Tunable`](@ref). Each switching time belongs to the
following interval. Supply switching times in sorted order. A `nothing` grid creates
one value. Time-based callbacks require
`initialization_times`, one per interval; `Parameter` supplies the problem start
followed by the switching times. `cache` optionally caches time lookups. Output
length and `bounds` are checked during Lux setup.
"""
function PiecewiseConstant(
        timegrid::Union{AbstractVector{<:Number}, Nothing}, default;
        name = nothing, cache = nothing, initialization_times = nothing,
        bounds = (-Inf, Inf), transform = nothing, prior = nothing, kwargs...
    )
    lookup = LookUp(timegrid, cache)
    n = isnothing(timegrid) ? 1 : length(timegrid) + 1
    isnothing(initialization_times) || length(initialization_times) == n ||
        throw(DimensionMismatch("Expected $n parameter initialization times."))
    transform = annotation_transform(transform, bounds...)
    initialize = (rng, n) -> initialize_annotation_values(default, rng, n, initialization_times, transform)
    return PiecewiseConstant(name, lookup, ParameterCollection(initialize, n, transform, prior))
end

function PiecewiseConstant(timegrid::Union{AbstractVector{<:Number}, Nothing}, parameters::NamedTuple; name = nothing, cache = nothing)
    return PiecewiseConstant(name, LookUp(timegrid, cache), parameters)
end

function (pc::PiecewiseConstant)(t, ps, st)
    (; lookup, parameter) = pc
    (idx, _), lookup_st = LuxCore.apply(lookup, t, ps.lookup, st.lookup)
    params, parameter_st = apply_scalar_collections(parameter, idx, ps.parameter, st.parameter)
    return params, (; lookup = lookup_st, parameter = parameter_st)
end

get_timestops(pc::PiecewiseConstant, ps, st, ::Type{T}) where {T} = get_timestops(pc.lookup, nothing, st.lookup, T)
get_timestops(pc::FixedLayer, ps, st, ::Type{T}) where {T} = get_timestops(pc.layer, st.parameters, st.states, T)

"""
$(TYPEDEF)

A collection of per-parameter piecewise-constant layers. Calling with `(t, ps, st)`
returns the reconstructed problem parameters and updated Lux state. Fixed values
are held in Lux state through [`FixedLayer`](@ref).

# Fields
$(FIELDS)
"""
@concrete terse struct Parameter <: LuxCore.AbstractLuxContainerLayer{(:parameters,)}
    "The optional name of the parameter collection."
    name
    "Reconstructs the problem parameters from their canonical tunable values."
    repack
    "Shared-grid parameter layers, keyed by the first member's symbol."
    parameters
    "Original parameter indices represented by each shared-grid layer."
    indices
end

Parameter(name, repack, parameters) = Parameter(name, repack, parameters, Tuple([i] for i in 1:length(parameters)))

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::Parameter)
    return (;
        parameters = map(child -> LuxCore.initialstates(rng, child), layer.parameters),
        permutation = collection_permutation(layer.indices),
    )
end

"""
$(SIGNATURES)

Construct parameter layers from `annotation => timegrid` definitions. Unspecified
parameters receive one tunable value initialized from the problem. `Constant`
annotations and entries selected by `fixed` store values in Lux state. `sys`
supplies symbols, defaults, and annotation bounds.
"""
function Parameter(
        prob::SciMLBase.AbstractSciMLProblem,
        raw_definitions...;
        name = nothing,
        fixed = [],
        sys = get_symbolic_container(prob),
    )
    definitions = reduce(
        vcat, map(raw_definitions) do definition
            process_definition(sys, Val(:parameter), definition...)
        end; init = Pair[]
    )
    indices = map(definition -> get_index(first(definition)), definitions)
    allunique(indices) || throw(ArgumentError("Define each parameter only once."))
    vals, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    symbols = SymbolicIndexingInterface.parameter_symbols(sys)
    fixed_indices = map(k -> k isa Integer ? k : SymbolicIndexingInterface.parameter_index(sys, k), fixed)
    entries = map(eachindex(vals)) do i
        sym = only(filter(s -> SymbolicIndexingInterface.parameter_index(sys, s) == i, symbols))
        position = findfirst(==(i), indices)
        if isnothing(position)
            grid, default, tunable, transform, prior = nothing, vals[i], true, BoundedTransform(), nothing
        else
            annotation, grid = definitions[position]
            default, tunable = get_default(annotation), is_tunable(annotation)
            transform, prior = get_transform(annotation), get_prior(annotation)
        end
        times = isnothing(grid) ? [first(prob.tspan)] : vcat([first(prob.tspan)], grid)
        n = length(times)
        initialize = (rng, n) -> initialize_annotation_values(default, rng, n, times, transform)
        parameter = ParameterCollection(initialize, n, transform, prior)
        fixed_value = !tunable || i in fixed_indices
        return (; index = i, name = Symbol(sym), grid, parameter, fixed = fixed_value)
    end
    groups = group_scalar_collections(entries)
    parameters = map(groups) do group
        entry = first(group)
        return shared_grid_layer(PiecewiseConstant, group, LookUp(entry.grid, nothing))
    end |> NamedTuple
    return Parameter(name, repack, parameters, Tuple([entry.index for entry in group] for group in groups))
end


function get_timestops(pc::Parameter, ps, st, ::Type{T} = Float64) where {T}
    tstops = reduce(
        vcat, map(keys(pc.parameters)) do id
            get_timestops(pc.parameters[id], nothing, st.parameters[id], T)
        end; init = T[]
    )
    sort!(tstops)
    unique!(tstops)
    return tstops
end

function (layer::Parameter)(t, ps, st)
    (; repack, parameters) = layer
    out, new_st = apply_parameters(parameters, t, ps.parameters, st.parameters, st.permutation)
    return repack(out), merge(st, (; parameters = new_st))
end

@generated function apply_parameters(params::NamedTuple{names}, t, ps, st::NamedTuple{names}, permutation) where {names}
    outs = Symbol[]
    sts = Symbol[]
    exprs = Expr[]
    for n in names
        push!(outs, gensym(:out))
        push!(sts, gensym(:st))
        push!(
            exprs,
            :(($(last(outs)), $(last(sts))) = params.$(n)(t, ps.$(n), st.$(n)))
        )
    end
    push!(
        exprs,
        :(new_st = NamedTuple{$(names)}(($(sts...),)))
    )
    # Children return scalars. A vector literal preserves scalar tracking;
    # ReverseDiff's vcat overload would instead track the array as a whole.
    push!(
        exprs,
        :(vals = $(canonical_output_expression(fieldtypes(params), outs)))
    )
    push!(exprs, :(return vals, new_st))
    return Expr(:block, exprs...)
end

(layer::Parameter)(prob::SciMLBase.AbstractDEProblem, ps, st) = layer(first(prob.tspan), ps, st)
(layer::Parameter)(sol::SciMLBase.AbstractTimeseriesSolution, ps, st) = layer(last(current_time(sol)), ps, st)

# Rebuild only changed parameter fields; extensions can preserve their own axes.
replace_parameter_fields(ps::NamedTuple, fields::NamedTuple) = merge(ps, fields)

function parameter_segment_layout(collection, inserted, times)
    added = filter(t -> isnothing(collection) || t ∉ collection, times)
    isempty(added) && return nothing
    grid = sort!(unique!(isnothing(collection) ? collect(added) : vcat(collection, added)))
    source = vcat(1, [isnothing(collection) ? 1 : searchsortedlast(collection, t) + 1 for t in grid])
    marked = [
        i + 1 for (i, t) in enumerate(grid)
            if t in added || (!isnothing(collection) && searchsortedlast(collection, t) + 1 in inserted)
    ]
    return grid, source, marked
end
ChainRulesCore.@non_differentiable parameter_segment_layout(collection, inserted, times)

function add_parameter_segments!(layer::PiecewiseConstant, ps, st, times)
    number_of_tunable_collections(layer.parameter) == 0 && return ps, st
    layout = parameter_segment_layout(st.lookup.collection, st.lookup.inserted, times)
    isnothing(layout) && return ps, st
    grid, source, inserted = layout
    parameters, parameter_state = expand_parameter_collections(layer.parameter, ps.parameter, st.parameter, source)
    next_ps = replace_parameter_fields(ps, (; parameter = parameters))
    lookup = merge(
        st.lookup, (;
            collection = grid, inserted,
            cache = isnothing(st.lookup.cache) ? nothing : empty(st.lookup.cache),
        )
    )
    return next_ps, merge(st, (; lookup, parameter = parameter_state))
end

add_parameter_segments!(::FixedLayer, ps, st, times) = (ps, st)

function add_parameter_segments!(layer::Parameter, ps, st, times)
    names = keys(layer.parameters)
    updated = map(names) do name
        add_parameter_segments!(layer.parameters[name], getproperty(ps.parameters, name), st.parameters[name], times)
    end
    parameters = NamedTuple{names}(map(first, updated))
    states = NamedTuple{names}(map(last, updated))
    return replace_parameter_fields(ps, (; parameters)), merge(st, (; parameters = states))
end


function expand_parameter_collections(::ParameterCollection, ps, st, source)
    return replace_parameter_fields(ps, (; ps = ps.ps[source])), st
end
function expand_parameter_collections(layer::FixedLayer, ps, st, source)
    parameters, state = expand_parameter_collections(layer.layer, st.parameters, st.states, source)
    return ps, merge(st, (; parameters, states = state))
end
function expand_parameter_collections(layers::NamedTuple, ps, st, source)
    names = keys(layers)
    updated = map(names) do name
        expand_parameter_collections(layers[name], getproperty(ps, name), st[name], source)
    end
    return NamedTuple{names}(map(first, updated)), NamedTuple{names}(map(last, updated))
end

# Time-grid adaptation. `plan` lists `(a, b, w)` per new interval: the new value
# is `v[a] + w * (v[b] - v[a])`, so constant sampling uses `w = 0`.
function resample_values(values, plan)
    return [w == 0 ? values[a] : values[a] + w * (values[b] - values[a]) for (a, b, w) in plan]
end

function resample_parameter_collections(collection::ParameterCollection, ps, st, f, name)
    return replace_parameter_fields(ps, (; ps = f(ps.ps, collection, name))), st
end
function resample_parameter_collections(layer::FixedLayer, ps, st, f, name)
    parameters, state = resample_parameter_collections(layer.layer, st.parameters, st.states, f, name)
    return ps, merge(st, (; parameters, states = state))
end
function resample_parameter_collections(layers::NamedTuple, ps, st, f, name)
    names = keys(layers)
    updated = map(names) do member
        resample_parameter_collections(layers[member], getproperty(ps, member), st[member], f, member)
    end
    return NamedTuple{names}(map(first, updated)), NamedTuple{names}(map(last, updated))
end

grid_lookup_state(::PiecewiseConstant, st) = st.lookup
grid_lookup_state(layer::FixedLayer, st) = grid_lookup_state(layer.layer, st.states)

# Replace one shared-grid group's lookup and resample all of its members.
# `f(values, plan, collection, member)` returns the member's new values.
function adapt_parameter_grid(pc::PiecewiseConstant, ps, st, grid, plan_for, f, name)
    plan = plan_for(st.lookup.collection)
    member_values = (values, collection, member) -> f(values, plan, collection, member)
    parameters, parameter_state = resample_parameter_collections(pc.parameter, ps.parameter, st.parameter, member_values, name)
    lookup = merge(
        st.lookup, (;
            collection = grid, inserted = Int[],
            cache = isnothing(st.lookup.cache) ? nothing : empty(st.lookup.cache),
        )
    )
    return replace_parameter_fields(ps, (; parameter = parameters)), merge(st, (; lookup, parameter = parameter_state))
end
function adapt_parameter_grid(layer::FixedLayer, ps, st, args...)
    parameters, state = adapt_parameter_grid(layer.layer, st.parameters, st.states, args...)
    return ps, merge(st, (; parameters, states = state))
end

number_of_parameter_constraints(layer::PiecewiseConstant, st) =
    number_of_tunable_collections(layer.parameter) * length(st.lookup.inserted)
number_of_parameter_constraints(::FixedLayer, st) = 0
function number_of_parameter_constraints(layer::Parameter, st)
    return sum(keys(layer.parameters); init = 0) do name
        number_of_parameter_constraints(layer.parameters[name], st.parameters[name])
    end
end
ChainRulesCore.@non_differentiable number_of_parameter_constraints(layer, st)

parameter_constraint_entry(::FixedLayer, ps, inserted) = (; fixed = true, values = nothing, inserted)
parameter_constraint_entry(::ParameterCollection, ps, inserted) = (; fixed = false, values = ps.ps, inserted)
function parameter_constraint_entries(layer::PiecewiseConstant, ps, st)
    layers = layer.parameter
    if layers isa NamedTuple
        return map(keys(layers)) do name
            parameter_constraint_entry(layers[name], getproperty(ps.parameter, name), st.lookup.inserted)
        end
    end
    return (parameter_constraint_entry(layers, ps.parameter, st.lookup.inserted),)
end
parameter_constraint_entries(::FixedLayer, ps, st) = ((; fixed = true, values = nothing, inserted = Int[]),)
function parameter_constraint_entries(layer::Parameter, ps, st)
    parts = map(keys(layer.parameters)) do name
        parameter_constraint_entries(layer.parameters[name], getproperty(ps.parameters, name), st.parameters[name])
    end
    flat = foldl((a, b) -> (a..., b...), parts; init = ())
    return map(i -> flat[i], st.permutation)
end

function append_parameter_constraints(layer::Parameter, ps, st, residuals)
    number_of_parameter_constraints(layer, st) == 0 && return residuals
    parts = map(parameter_constraint_entries(layer, ps, st)) do entry
        entry.fixed ? similar(residuals, 0) :
            [entry.values[i] - entry.values[i - 1] for i in entry.inserted]
    end
    return vcat(residuals, parts...)
end

function write_parameter_constraints!(res, index, layer::Parameter, ps, st)
    for entry in parameter_constraint_entries(layer, ps, st)
        entry.fixed && continue
        for i in entry.inserted
            res[index] = entry.values[i] - entry.values[i - 1]
            index += 1
        end
    end
    return index
end

active_parameter_columns(::FixedLayer, ids, st, t) = BitSet()
function active_parameter_columns(layer::PiecewiseConstant, ids, st, t)
    position = isnothing(st.lookup.collection) ? 1 : searchsortedlast(st.lookup.collection, t) + 1
    if layer.parameter isa NamedTuple
        columns = BitSet()
        for name in keys(layer.parameter)
            layer.parameter[name] isa FixedLayer && continue
            push!(columns, Int(getproperty(ids.parameter, name).ps[position]))
        end
        return columns
    end
    return BitSet([Int(ids.parameter.ps[position])])
end
function active_parameter_columns(layer::Parameter, ids, st, t)
    columns = BitSet()
    for name in keys(layer.parameters)
        union!(columns, active_parameter_columns(layer.parameters[name], getproperty(ids.parameters, name), st.parameters[name], t))
    end
    return columns
end

# Resolve only the selected signal. Build the canonical index-to-group mapping
# once so repeated queries do not scan groups or materialize unused dependencies.
function parameter_signal_locations(layer::Parameter, st)
    locations = Tuple{Symbol, Union{Nothing, Symbol}}[]
    sizehint!(locations, sum(length, layer.indices; init = 0))
    for (group, name) in enumerate(keys(layer.parameters))
        child = layer.parameters[name]
        grouped = !(child isa FixedLayer) && child.parameter isa NamedTuple
        for position in eachindex(layer.indices[group])
            member = grouped ? keys(child.parameter)[position] : nothing
            push!(locations, (name, member))
        end
    end
    return locations[st.permutation]
end
function parameter_signal_dependency(layer::Parameter, ids, st, location, t)
    name, member = location
    child = layer.parameters[name]
    child isa FixedLayer && return BitSet()
    parameters = getproperty(ids.parameters, name).parameter
    if !isnothing(member)
        child.parameter[member] isa FixedLayer && return BitSet()
        parameters = getproperty(parameters, member)
    end
    grid = st.parameters[name].lookup.collection
    intervals = if t isa Tuple
        lookup!(nothing, grid, first(t)):lookup!(nothing, grid, last(t))
    else
        interval = lookup!(nothing, grid, t)
        interval:interval
    end
    return BitSet(Int(parameters.ps[i]) for i in intervals)
end
