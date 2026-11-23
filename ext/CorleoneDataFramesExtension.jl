module CorleoneDataFramesExtension

import Corleone
using Corleone: Experiment, Trajectory, Observation
import DataFrames: AbstractDataFrame, DataFrame, groupby, nrow
import SymbolicIndexingInterface: variable_symbols, symbolic_container, variable_index

blank(x) = ismissing(x) || isnothing(x) || x == ""

function identifier(x, column)
    x isa Union{AbstractString, Symbol} && !isempty(string(x)) ||
        throw(ArgumentError("$column must contain nonempty string or symbol IDs."))
    return string(x)
end

function lookup(mapping, id)
    haskey(mapping, Symbol(id)) && return mapping[Symbol(id)]
    if !(mapping isa NamedTuple) && haskey(mapping, id)
        return mapping[id]
    end
    throw(ArgumentError("No mapping supplied for ID `$id`."))
end

function noise_scale(x)
    blank(x) && return 1.0
    value = x isa AbstractString ? tryparse(Float64, x) : x
    value isa Real && isfinite(value) && value > 0 ||
        throw(ArgumentError("noiseParameters must contain positive numeric standard deviations; parameter IDs and noise formulas require full PEtab support."))
    return value
end

function Corleone.experiments_from_dataframe(
        df::AbstractDataFrame; observables = nothing, initial_states = nothing
    )
    required = (:observableId, :experimentId, :time, :measurement)
    all(in(propertynames(df)), required) ||
        throw(ArgumentError("Expected PEtab 2 columns observableId, experimentId, time, measurement."))
    nrow(df) > 0 || throw(ArgumentError("Supply at least one measurement."))
    for column in (:observableParameters, :preequilibrationConditionId, :simulationConditionId)
        if column in propertynames(df) && any(x -> !blank(x), df[!, column])
            throw(ArgumentError("$column is unsupported; supply observable mappings and experiment configuration separately."))
        end
    end
    all(t -> t isa Real && isfinite(t), df.time) ||
        throw(ArgumentError("Measurement times must be finite real numbers; steady-state measurements are unsupported."))
    all(y -> y isa Real && isfinite(y), df.measurement) ||
        throw(ArgumentError("Measurements must be finite real numbers."))
    # Normalize string and symbol identifiers without modifying the input table.
    ids = identifier.(df.experimentId, Ref(:experimentId))
    observable_ids = identifier.(df.observableId, Ref(:observableId))
    normalized = DataFrame(
        experimentId = ids, observableId = observable_ids,
        time = df.time, measurement = df.measurement,
    )
    has_noise = :noiseParameters in propertynames(df)
    scales = has_noise ? noise_scale.(df.noiseParameters) : nothing
    has_noise && (normalized[!, :noiseParameters] = scales)
    names = Symbol[]
    experiments = Experiment[]
    for experiment in groupby(normalized, :experimentId; sort = false)
        id = first(experiment.experimentId)
        observations = Observation[]
        for observable in groupby(experiment, :observableId; sort = false)
            observable_id = first(observable.observableId)
            selector = isnothing(observables) ? Symbol(observable_id) : lookup(observables, observable_id)
            # Put the nth measurement at each time into lane n. Each lane has
            # unique sorted times while all replicates remain in the objective.
            counts = Dict{Any, Int}()
            lanes = Vector{Int}[]
            for i in sortperm(observable.time)
                t = observable.time[i]
                lane = get(counts, t, 0) + 1
                counts[t] = lane
                lane > length(lanes) && push!(lanes, Int[])
                push!(lanes[lane], i)
            end
            for indices in lanes
                data = (observable.time[indices], observable.measurement[indices])
                has_noise && (data = (data..., observable.noiseParameters[indices]))
                push!(observations, Observation(selector => data))
            end
        end
        u0 = isnothing(initial_states) ? nothing : lookup(initial_states, id)
        times = sort!(unique!(reduce(vcat, map(o -> o.t, observations))))
        push!(names, Symbol(id))
        push!(experiments, Experiment(Tuple(observations), u0, times))
    end
    return NamedTuple{Tuple(names)}(Tuple(experiments))
end

function Corleone.Experiment(df::AbstractDataFrame; observables = nothing, u0 = nothing)
    experiments = Corleone.experiments_from_dataframe(df; observables)
    length(experiments) == 1 || throw(ArgumentError("Experiment(df) requires exactly one experimentId; use experiments_from_dataframe for multiple experiments."))
    experiment = only(values(experiments))
    return Experiment(experiment.observations, u0, experiment.t)
end

function output_observables(traj, observables)
    mappings = isnothing(observables) ?
        [string(sym) => variable_index(traj, sym) for sym in variable_symbols(symbolic_container(traj))] :
        collect(Base.pairs(observables))
    # Also accept a vector/tuple of ID => selector pairs.
    if observables isa Union{AbstractVector, Tuple} && !(observables isa NamedTuple)
        mappings = collect(observables)
    end
    all(p -> p isa Pair, mappings) || throw(ArgumentError("observables must map IDs to selectors."))
    ids = [identifier(first(p), :observableId) for p in mappings]
    allunique(ids) || throw(ArgumentError("Output observable IDs must be unique."))
    return [id => last(p) for (id, p) in zip(ids, mappings)]
end

# A function barrier specializes each callback independently, including callable
# structs, so sample collection need not start from an untyped accumulator.
function simulation_values(traj, selector)
    selector isa Expr && throw(ArgumentError("Export selectors must be state symbols, indices, or scalar callbacks."))
    resolved = if selector isa Symbol
        index = variable_index(traj, selector)
        isnothing(index) && throw(ArgumentError("Unknown trajectory state `$selector`."))
        index
    else
        selector
    end
    return map(eachindex(traj.t)) do k
        value = Corleone.observe(resolved, traj, k)
        value isa Real || throw(ArgumentError("Export observables must return real scalars."))
        return value
    end
end

function Corleone.trajectories_dataframe(
        traj::Trajectory; experiment_id = "experiment1", observables = nothing
    )
    id = identifier(experiment_id, :experimentId)
    mappings = output_observables(traj, observables)
    # vcat promotes different numeric result types, preserving e.g. BigFloat or
    # dual-valued callbacks rather than converting them to the saved state type.
    columns = map(mapping -> simulation_values(traj, last(mapping)), mappings)
    simulations = isempty(columns) ? eltype(first(traj.u))[] : reduce(vcat, columns)
    n, m = length(traj.t), length(mappings)
    outgoing = Set(traj.reset_indices)
    incoming = Set(traj.reset_indices .- 1)
    sides = map(eachindex(traj.t)) do k
        k in outgoing ? "outgoing" : k in incoming ? "incoming" : "none"
    end
    return DataFrame(
        observableId = repeat(first.(mappings); inner = n), experimentId = fill(id, n * m),
        time = repeat(traj.t, m), simulation = simulations,
        sampleIndex = repeat(collect(eachindex(traj.t)), m), resetSide = repeat(sides, m),
    )
end

function Corleone.trajectories_dataframe(
        trajectories::Union{AbstractVector, Tuple, NamedTuple}; experiment_ids = nothing, observables = nothing
    )
    all(t -> t isa Trajectory, values(trajectories)) || throw(ArgumentError("Expected a collection of Trajectories."))
    isempty(trajectories) && throw(ArgumentError("Supply at least one trajectory."))
    ids = isnothing(experiment_ids) ?
        (trajectories isa NamedTuple ? string.(keys(trajectories)) : ["experiment$i" for i in 1:length(trajectories)]) :
        collect(experiment_ids)
    length(ids) == length(trajectories) || throw(DimensionMismatch("Supply one experiment ID per trajectory."))
    ids = identifier.(ids, Ref(:experimentId))
    allunique(ids) || throw(ArgumentError("Each trajectory must have a unique experiment ID."))
    frames = [
        Corleone.trajectories_dataframe(traj; experiment_id = id, observables)
            for (traj, id) in zip(values(trajectories), ids)
    ]
    return reduce(vcat, frames)
end

DataFrame(traj::Trajectory; kwargs...) = Corleone.trajectories_dataframe(traj; kwargs...)
DataFrame(trajectories::AbstractVector{<:Trajectory}; kwargs...) =
    Corleone.trajectories_dataframe(trajectories; kwargs...)

end
