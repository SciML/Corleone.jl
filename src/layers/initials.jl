"""
$(TYPEDEF)
"""
@concrete terse struct InitialCondition <: LuxCore.AbstractLuxContainerLayer{(:tunable, :prescribed)}
    name
    tunable
    prescribed
    repack
    tunable_indices
    permutation
end

function InitialCondition(problem::SciMLBase.AbstractSciMLProblem, definitions...; name = nothing, fixed = [])
    sys = get_symbolic_container(problem)
    definitions = reduce(
        vcat, map(definitions) do def
            process_definition(sys, Val(:variable), def...)
        end; init = Pair[]
    )
    missing_defs = setdiff(
        [SymbolicIndexingInterface.variable_index(sys, k) for k in SymbolicIndexingInterface.variable_symbols(sys)],
        first.(definitions)
    )
    vals, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), problem.u0)
    missing_defs = [k => vals[k] for k in missing_defs]
    definitions = vcat(definitions, missing_defs)
    sort!(definitions, by = first)
    fixed = map(k -> k isa Integer ? k : SymbolicIndexingInterface.variable_index(sys, k), fixed)
    free = setdiff(first.(definitions), fixed)
    values = last.(definitions)
    perm = sortperm(vcat(free, fixed))
    vsymmap = Dict([SymbolicIndexingInterface.variable_index(sys, k) => k for k in SymbolicIndexingInterface.variable_symbols(sys)])
    free_values = map(free) do i
        (Symbol(vsymmap[i]), values[i])
    end |> NamedTuple
    fixed_values = map(fixed) do i
        (Symbol(vsymmap[i]), values[i])
    end |> NamedTuple

    return InitialCondition(
        name,
        free_values, fixed_values, repack, sort!(free), perm
    )
end

LuxCore.initialparameters(::Random.AbstractRNG, x::InitialCondition) = deepcopy(x.tunable)
LuxCore.initialstates(::Random.AbstractRNG, x::InitialCondition) = deepcopy(x.prescribed)

function (layer::InitialCondition)(default, ps, st)
    (; repack, permutation, tunable_indices) = layer
    vals = vcat(collect(values(ps)), collect(values(st)))
    ordered = vals[permutation]
    residuals = isempty(tunable_indices) ? similar(ordered, 0) : [ordered[i] - default[i] for i in tunable_indices]
    return (repack(ordered), residuals), st
end

@concrete terse struct InitialConditions <: LuxCore.AbstractLuxContainerLayer{(:lookup, :initialconditions)}
    name
    lookup
    initialconditions
end

LuxCore.initialparameters(rng::Random.AbstractRNG, layer::InitialConditions) = (;
    lookup = LuxCore.initialparameters(rng, layer.lookup),
    initialconditions = map(layer.initialconditions) do ic
        LuxCore.initialparameters(rng, ic)
    end,
)

LuxCore.initialstates(rng::Random.AbstractRNG, layer::InitialConditions) = (;
    lookup = LuxCore.initialstates(rng, layer.lookup),
    initialconditions = map(layer.initialconditions) do ic
        LuxCore.initialstates(rng, ic)
    end,
)

function InitialConditions(
        problem::SciMLBase.AbstractSciMLProblem, definitions...;
        name = nothing,
    )
    timepoints = first.(definitions)
    inner_defs = last.(definitions)
    lookup = LookUp(
        Dict{eltype(problem.tspan), UInt64}([k => i  for (i, k) in enumerate(timepoints)]), nothing
    )
    conditions = map(inner_defs) do def
        InitialCondition(
            problem,
            (haskey(def, :variables) ? def.variables : ())...;
            fixed = haskey(def, :fixed) ? def.fixed : [],
            name = haskey(def, :name) ? def.name : nothing,
        )
    end
    return InitialConditions(name, lookup, conditions)
end

get_timestops(ic::InitialConditions, ps, st, ::Type{T}) where {T} = get_timestops(ic.lookup, ps.lookup, st.lookup, T)

istypemax(x::V) where {V} = x === typemax(V)

function (layer::InitialConditions)((default, t0)::Tuple, ps, st)
    (; lookup, initialconditions) = layer
    (id, _), st_lookup = LuxCore.apply(lookup, t0, ps.lookup, st.lookup)
    istypemax(id) && return (default, similar(default, 0)), (; lookup = st_lookup, initialconditions = st.initialconditions)
    (newval, shooting), _ = initialconditions[id](default, ps.initialconditions[id], st.initialconditions[id])
    return (newval, shooting), (; lookup = st_lookup, initialconditions = st.initialconditions)
end

#(layer::InitialConditions)(prob::SciMLBase.AbstractDEProblem, ps, st) = layer((prob.u0, first(prob.tspan)), ps, st)
#(layer::InitialConditions)(sol::SciMLBase.AbstractTimeseriesSolution, ps, st) = layer((last(state_values(sol)), last(current_time(sol))), ps, st)
