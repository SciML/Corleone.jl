"""
$(TYPEDEF)

A single piecewise-constant parameter. Lookup times separate intervals, so a grid
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
    "The collection of parameter values, one per interval."
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
function PiecewiseConstant(timegrid::Union{AbstractVector{<:Number}, Nothing}, default;
        name = nothing, cache = nothing, initialization_times = nothing,
        bounds = (-Inf, Inf), kwargs...)
    lookup = LookUp(timegrid, cache)
    n = isnothing(timegrid) ? 1 : length(timegrid) + 1
    isnothing(initialization_times) || length(initialization_times) == n ||
        throw(DimensionMismatch("Expected $n parameter initialization times."))
    initialize = (rng, n) -> initialize_annotation_values(default, rng, n, initialization_times, bounds)
    return PiecewiseConstant(name, lookup, ParameterCollection(initialize, n))
end

function (pc::PiecewiseConstant)(t, ps, st)
    (; lookup, parameter) = pc
    (idx, inserted), lookup_st = LuxCore.apply(lookup, t, ps.lookup, st.lookup)
    params, parameter_st = LuxCore.apply(parameter, idx, ps.parameter, st.parameter)
    return params, (; lookup = lookup_st, parameter = parameter_st)
end

get_timestops(pc::PiecewiseConstant, ps, st, ::Type{T}) where {T} = get_timestops(pc.lookup, ps.lookup, st.lookup, T)
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
    "A NamedTuple of per-parameter layers in canonical parameter index order."
    parameters
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
        defintions...;
        name = nothing,
        fixed = [],
        sys = get_symbolic_container(prob),
    )
    definitions = reduce(vcat, map(defintions) do definition
        process_definition(sys, Val(:parameter), definition...)
    end; init = Pair[])
    indices = map(definition -> get_index(first(definition)), definitions)
    allunique(indices) || throw(ArgumentError("Define each parameter only once."))
    vals, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    symbols = SymbolicIndexingInterface.parameter_symbols(sys)
    fixed_indices = map(k -> k isa Integer ? k : SymbolicIndexingInterface.parameter_index(sys, k), fixed)
    parameters = map(eachindex(vals)) do i
        sym = only(filter(s -> SymbolicIndexingInterface.parameter_index(sys, s) == i, symbols))
        position = findfirst(==(i), indices)
        if isnothing(position)
            layer = PiecewiseConstant(nothing, vals[i]; name = sym)
            tunable = true
        else
            annotation, grid = definitions[position]
            times = isnothing(grid) ? [first(prob.tspan)] : vcat([first(prob.tspan)], grid)
            layer = PiecewiseConstant(grid, get_default(annotation); name = sym,
                initialization_times = times, bounds = get_bounds(annotation))
            tunable = is_tunable(annotation)
        end
        Symbol(sym) => (tunable && i ∉ fixed_indices ? layer : FixedLayer(layer))
    end |> NamedTuple
    return Parameter(name, repack, parameters)
end


function get_timestops(pc::Parameter, ps, st, ::Type{T} = Float64) where {T}
    tstops = reduce(
        vcat, map(keys(pc.parameters)) do id
            get_timestops(pc.parameters[id], ps.parameters[id], st.parameters[id], T)
        end; init = T[]
    )
    sort!(tstops)
    unique!(tstops)
    return tstops
end

function (layer::Parameter)(t, ps, st)
    (; repack, parameters) = layer
    out, new_st = apply_parameters(parameters, t, ps.parameters, st.parameters)
    return repack(out), (; parameters = new_st)
end

@generated function apply_parameters(params::NamedTuple{names}, t, ps, st::NamedTuple{names}) where {names}
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
    push!(
        exprs,
        :(vals = vcat($(outs...)))
    )
    push!(exprs, :(return vals, new_st))
    return Expr(:block, exprs...)
end

(layer::Parameter)(prob::SciMLBase.AbstractDEProblem, ps, st) = layer(first(prob.tspan), ps, st)
(layer::Parameter)(sol::SciMLBase.AbstractTimeseriesSolution, ps, st) = layer(last(current_time(sol)), ps, st)
