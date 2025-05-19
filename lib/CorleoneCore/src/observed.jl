"""
$(TYPEDEF)

A collection of [`ObservedFunction`](@ref).

# Fields
$(FIELDS)
"""
struct ObservedFunctions{O,S,A}
    "The functions"
    observed::O
    "The time points which are expected to be present in the [`Trajectory`](@ref)"
    saveats::S
    "The aggregation of all functions, is `reduce(vcat, obs...)` per default."
    aggregation::A
end



#TODO Think about using a NamedTuple here to allow for complex aggregations
function ObservedFunctions(container, specs::NamedTuple...; aggregation=Base.Fix1(reduce, vcat))
    # Unpack the first named tuple
    saveats = reduce(vcat, map(specs) do spec
        spec.saveats
    end)
    sort!(saveats)
    unique!(saveats)

    obs = map(specs) do spec
        indices = findall(∈(spec.saveats), saveats)
        obsfun = SymbolicIndexingInterface.observed(container, spec.expression)
        merge(spec, (; observed=obsfun, indices=indices))
    end

    return ObservedFunctions{typeof(obs),typeof(saveats),typeof(aggregation)}(obs, saveats, aggregation)
end

function apply_observed(specs::NamedTuple, trajectory::Trajectory)
    (; observed, indices) = specs
    map(indices) do i
        @views observed(trajectory.states[:, i], trajectory.parameters, trajectory.time[i])
    end
end

function (obs::ObservedFunctions)(trajectory::Trajectory)
    (; observed, aggregation, saveats) = obs
    aggregation(map(observed) do spec
        apply_observed(spec, trajectory)
    end)
end

(obs::ObservedFunctions)(sol) = obs(Trajectory(sol))
