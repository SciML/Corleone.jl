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
    "The collection of values assigned at the scheduled reset times."
    parameter
end

"""
$(SIGNATURES)

Construct reset values for one state variable on `timegrid`. `default` is a scalar,
vector, or count/timepoint initializer as described by [`Tunable`](@ref). Times must
be unique; `nothing` selects an empty reset grid. `cache` is an optional lookup
cache. Output length and `bounds` are checked during Lux setup.
"""
function InitialCondition(timegrid::AbstractVector{<:Real}, default;
        name = nothing, cache = nothing, bounds = (-Inf, Inf))
    allunique(timegrid) || throw(ArgumentError("Initial-condition times must be unique."))
    lookup = LookUp(Dict(t => i for (i, t) in enumerate(timegrid)), cache)
    initialize = (rng, n) -> initialize_annotation_values(default, rng, n, timegrid, bounds)
    return InitialCondition(name, lookup, ParameterCollection(initialize, length(timegrid)))
end

function InitialCondition(::Nothing, default; kwargs...)
    return InitialCondition(Float64[], default; kwargs...)
end


function LuxCore.initialstates(rng::Random.AbstractRNG, x::InitialCondition)
    (;
        lookup=LuxCore.initialstates(rng, x.lookup),
        parameter=LuxCore.initialstates(rng, x.parameter),
        active=false
    )

end

function (layer::InitialCondition)((incoming, t)::Tuple, ps, st)
    T = promote_type(typeof(incoming), eltype(ps.parameter.ps))
    haskey(layer.lookup, t, ps.lookup, st.lookup) || return T(incoming), merge(st, (; active=false))
    (index, _), lookup_st = LuxCore.apply(layer.lookup, t, ps.lookup, st.lookup)
    value, parameter_st = LuxCore.apply(layer.parameter, index, ps.parameter, st.parameter)
    return value, (; lookup=lookup_st, parameter=parameter_st, active=true)
end

istypemax(x::V) where {V} = x === typemax(V)

get_timestops(layer::InitialCondition, ps, st, ::Type{T}) where {T} =
    get_timestops(layer.lookup, nothing, st.lookup, T)

"""
$(TYPEDEF)

A collection of single-variable `InitialCondition` layers, analogous to `Parameter`.
Definitions are `annotation => timegrid` pairs; plain indices and symbols are also
accepted as tunable annotations. Variables with no grid pass through their incoming
values. `Constant` annotations store scheduled values in Lux state via `FixedLayer`.

Calling with `(incoming_state, t)` returns `((state, residuals), next_st)`. Residuals
are assigned values minus the corresponding incoming values, in state index
order. Only active, non-fixed layers contribute residuals.
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
    "A NamedTuple of per-variable reset or pass-through layers in state index order."
    initialconditions
end

function LuxCore.initialstates(rng::Random.AbstractRNG, layer::InitialConditions)
    return (;
        initialconditions=map(ic -> LuxCore.initialstates(rng, ic), layer.initialconditions),
        active=fill(false, length(layer.initialconditions)),
    )
end

"""
$(SIGNATURES)

Construct per-variable initial conditions from `annotation => timegrid` definitions.
Annotations of kind `Val(:variable)` select state entries. Unspecified variables
pass through their incoming values; `Constant` definitions store reset values in
Lux state. `sys` supplies symbols, defaults, and annotation bounds.
"""
function InitialConditions(problem::SciMLBase.AbstractSciMLProblem, definitions...;
        name=nothing, sys=get_symbolic_container(problem))
    vals, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), problem.u0)
    processed = reduce(vcat, map(definitions) do definition
        process_definition(sys, Val(:variable), definition...)
    end; init=Pair[])
    indices = map(definition -> get_index(first(definition)), processed)
    allunique(indices) || throw(ArgumentError("Define each initial-condition variable only once."))
    symbols = SymbolicIndexingInterface.variable_symbols(sys)
    conditions = map(eachindex(vals)) do i
        position = findfirst(==(i), indices)
        sym = only(filter(s -> SymbolicIndexingInterface.variable_index(sys, s) == i, symbols))
        if isnothing(position)
            layer = PassThrough(sym)
        else
            annotation, grid = processed[position]
            layer = InitialCondition(grid, get_default(annotation); name=sym, bounds=get_bounds(annotation))
            layer = is_tunable(annotation) ? layer : FixedLayer(layer)
        end
        return Symbol(sym) => layer
    end |> NamedTuple
    return InitialConditions(name, repack, conditions)
end

function get_timestops(layer::InitialConditions, ps, st, ::Type{T}) where {T}
    times = reduce(vcat, map(keys(layer.initialconditions)) do key
        get_timestops(layer.initialconditions[key], nothing, st.initialconditions[key], T)
    end; init=T[])
    return sort!(unique!(times))
end


is_active(::InitialCondition, st) = st.active

"""
$(SIGNATURES)

Update the preallocated residual activity mask from per-variable layers and their
new Lux states. Only active, non-fixed initial conditions contribute residuals.
This scheduling update is non-differentiable; an all-`PassThrough` tuple clears
the mask without checking individual layers.
"""
function update_active_ics!(active, initial_conditions::Tuple, new_st::Tuple)
    foreach(initial_conditions, new_st, eachindex(active)) do ic, state, i
        active[i] = ic isa InitialCondition && is_active(ic, state)
    end
    return active
end

# These entries are initialized false and can never become active.
update_active_ics!(active, ::Tuple{Vararg{PassThrough}}, ::Tuple) = active .= false

ChainRulesCore.@non_differentiable update_active_ics!(active, initial_conditions, new_st)

function (layer::InitialConditions)((incoming, t)::Tuple, ps, st)
    out, new_st = apply_ics(layer.initialconditions, incoming, t, ps.initialconditions, st.initialconditions)
    active = update_active_ics!(st.active, values(layer.initialconditions), values(new_st))
    residuals = [out[i] - incoming[i] for i in eachindex(out) if active[i]]
    return (layer.repack(out), residuals), (; initialconditions=new_st, active)
end

@generated function apply_ics(ics::NamedTuple{names}, vals, t, ps, st::NamedTuple{names}) where names
    outs = Symbol[]
    sts = Symbol[]
    exprs = Expr[]
    for (i, n) in enumerate(names)
        push!(outs, gensym(:out))
        push!(sts, gensym(:st))
        push!(
            exprs,
            :(($(last(outs)), $(last(sts))) = ics.$(n)((vals[$(i)], t), ps.$(n), st.$(n)))
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
