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
following interval. A `nothing` grid creates one value. Time-based callbacks require
`initialization_times`, one per interval; `Parameter` supplies the problem start
followed by the switching times. `cache` optionally caches time lookups. Output
length and `bounds` are checked during Lux setup.
"""
function PiecewiseConstant(
        timegrid::Union{AbstractVector{<:Number}, Nothing}, default;
        name = nothing, cache = nothing, initialization_times = nothing,
        bounds = (-Inf, Inf), kwargs...
    )
    lookup = LookUp(timegrid, cache)
    n = isnothing(timegrid) ? 1 : length(timegrid) + 1
    isnothing(initialization_times) || length(initialization_times) == n ||
        throw(DimensionMismatch("Expected $n parameter initialization times."))
    initialize = (rng, n) -> initialize_annotation_values(default, rng, n, initialization_times, bounds)
    return PiecewiseConstant(name, lookup, ParameterCollection(initialize, n))
end

function PiecewiseConstant(timegrid::Union{AbstractVector{<:Number}, Nothing}, parameters::NamedTuple; name = nothing, cache = nothing)
    return PiecewiseConstant(name, LookUp(timegrid, cache), parameters)
end

function (pc::PiecewiseConstant)(t, ps, st)
    (; lookup, parameter) = pc
    (idx, inserted), lookup_st = LuxCore.apply(lookup, t, ps.lookup, st.lookup)
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
    "Original indices of the variables represented by each shared-grid layer."
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
            grid, default, bounds, tunable = nothing, vals[i], (-Inf, Inf), true
        else
            annotation, grid = definitions[position]
            default, bounds, tunable = get_default(annotation), get_bounds(annotation), is_tunable(annotation)
        end
        times = isnothing(grid) ? [first(prob.tspan)] : vcat([first(prob.tspan)], grid)
        n = length(times)
        initialize = (rng, n) -> initialize_annotation_values(default, rng, n, times, bounds)
        parameter = ParameterCollection(initialize, n)
        fixed_value = !tunable || i in fixed_indices
        return (; index = i, name = Symbol(sym), grid, parameter, fixed = fixed_value)
    end
    groups = group_scalar_collections(entries)
    parameters = map(groups) do group
        entry = first(group)
        if length(group) == 1
            pc = PiecewiseConstant(entry.name, LookUp(entry.grid, nothing), entry.parameter)
            entry.name => (entry.fixed ? FixedLayer(pc) : pc)
        else
            children = NamedTuple(Tuple(item.name => (item.fixed ? FixedLayer(item.parameter) : item.parameter) for item in group))
            entry.name => PiecewiseConstant(entry.grid, children; name = entry.name)
        end
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
