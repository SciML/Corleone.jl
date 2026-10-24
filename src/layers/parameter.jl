"""
$(TYPEDEF)
"""
@concrete terse struct PiecewiseConstant <: LuxCore.AbstractLuxContainerLayer{(:lookup, :parameter)}
    ""
    name
    ""
    lookup
    ""
    parameter
end

function PiecewiseConstant(timegrid::AbstractVector{<:Number}, default; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(timegrid, cache)
    parameters = ParameterCollection((rng, outdims) -> fill(default, outdims), length(timegrid) + 1)
    return PiecewiseConstant(name, lookup, parameters)
end

function PiecewiseConstant(timegrid::AbstractVector{<:Number}, default::Base.Callable; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(timegrid, cache)
    parameters = ParameterCollection(default, length(timegrid) + 1)
    return PiecewiseConstant(name, lookup, parameters)
end

function PiecewiseConstant(timegrid::Nothing, default::Base.Callable; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(timegrid, cache)
    parameters = ParameterCollection(default, 1)
    return PiecewiseConstant(name, lookup, parameters)
end

function PiecewiseConstant(timegrid::Nothing, default; name = nothing, cache = nothing, kwargs...)
    lookup = LookUp(timegrid, cache)
    parameters = ParameterCollection((rng, outdims) -> fill(default, outdims), 1)
    return PiecewiseConstant(name, lookup, parameters)
end

function (pc::PiecewiseConstant)(t, ps, st)
    (; lookup, parameter) = pc
    (idx, inserted), lookup_st = LuxCore.apply(lookup, t, ps.lookup, st.lookup)
    params, parameter_st = LuxCore.apply(parameter, idx, ps.parameter, st.parameter)
    return params, (; lookup = lookup_st, parameter = parameter_st)
end

get_timestops(pc::PiecewiseConstant, ps, st, ::Type{T}) where {T} = get_timestops(pc.lookup, ps.lookup, st.lookup, T)
get_timestops(pc::FixedLayer, ps, st, ::Type{T}) where {T} = get_timestops(pc.layer, st.parameters, st.states, T)

@concrete terse struct Parameter <: LuxCore.AbstractLuxContainerLayer{(:parameters,)}
    ""
    name
    ""
    repack
    ""
    parameters
end

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
            layer = PiecewiseConstant(grid, get_default(annotation); name = sym)
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
