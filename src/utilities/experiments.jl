"""
$(TYPEDEF)

Measurements `y` of one observed quantity at the sorted, unique times `t`. The
`selector` is a state symbol or index, or a map `(u, p, t) -> y` returning a scalar.
The noise scale `σ` is `nothing` (unweighted residuals), a scalar, or a vector
matching `y`.

# Fields
$(FIELDS)
"""
@concrete terse struct Observation
    "The observed state symbol or index, or a scalar map `(u, p, t) -> y`."
    selector
    "Sorted, unique observation times."
    t
    "The measured values."
    y
    "The noise scale, or `nothing` for unweighted residuals."
    σ
end

"""
$(SIGNATURES)

Construct an [`Observation`](@ref) from `selector => (t, y)` or `selector => (t, y, σ)`.
"""
function Observation((selector, data)::Pair)
    length(data) in (2, 3) || throw(ArgumentError("Observation data must contain (t, y) or (t, y, σ)."))
    t, y, σ = length(data) == 2 ? (data..., nothing) : data
    length(t) == length(y) || throw(DimensionMismatch("Expected $(length(t)) values for $selector, got $(length(y))."))
    issorted(t) && allunique(t) || throw(ArgumentError("Observation times of $selector must be sorted and unique."))
    isnothing(σ) || (σ isa Real ? σ > 0 : all(>(0), σ)) || throw(ArgumentError("Noise scales of $selector must be positive."))
    isnothing(σ) || σ isa Real || length(σ) == length(y) ||
        throw(DimensionMismatch("Noise scales of $selector must be scalar or match its values."))
    return Observation(selector, collect(t), collect(float.(y)), σ)
end

"""
$(TYPEDEF)

The observations of one experiment. An initial state `u0` replaces the problem's
initial values of all states that are not tunable.

# Fields
$(FIELDS)
"""
@concrete terse struct Experiment
    "The observations, one per observed quantity."
    observations
    "The initial state, or `nothing` to use the problem's initial state."
    u0
    "The sorted union of all observation times."
    t
end

"""
$(SIGNATURES)

Construct an [`Experiment`](@ref) from `selector => (t, y)` or `selector => (t, y, σ)`
pairs, e.g. `Experiment(:u₁ => (t₁, y₁), :u₂ => (t₂, y₂, 0.1); u0 = [1.0, 2.0])`.
"""
function Experiment(observations::Pair...; u0 = nothing)
    isempty(observations) && throw(ArgumentError("Supply at least one observation."))
    observations = map(Observation, observations)
    t = sort!(unique!(reduce(vcat, map(observation -> observation.t, observations))))
    return Experiment(observations, u0, t)
end

is_weighted(experiment::Experiment) = map(observation -> !isnothing(observation.σ), experiment.observations)
n_observations(experiment::Experiment) = sum(observation -> length(observation.y), experiment.observations)

"""
$(SIGNATURES)

Return the Lux state `st` of `layer` with the observation times of `experiment`
merged into its segment grid, so that predictions sample every observation time.
"""
function experiment_state(layer::ShootingLayer, experiment::Experiment, st)
    t0, tf = layer.problem.tspan
    all(t -> t0 <= t <= tf, experiment.t) ||
        throw(ArgumentError("Observation times must lie within the problem time span."))
    isnothing(experiment.u0) || length(experiment.u0) == length(layer.problem.u0) ||
        throw(DimensionMismatch("An experiment's u0 must match the problem's state length."))
    return merge(st, (; timestops = sort!(unique!(vcat(st.timestops, experiment.t)))))
end

"""
$(SIGNATURES)

Predict `experiment` with the shooting layer: simulate from the experiment's initial
state on the segment grid in `st`, prepared by [`experiment_state`](@ref).
"""
function (layer::ShootingLayer)(experiment::Experiment, ps, st)
    return layer((experiment_problem(layer, experiment), st.timestops), ps, st)
end

experiment_problem(layer, experiment) =
    isnothing(experiment.u0) ? layer.problem : remake(layer.problem; u0 = experiment.u0)
ChainRulesCore.@non_differentiable experiment_problem(::Any, ::Any)

"""
$(SIGNATURES)

Return the observation residuals `(y - ŷ) / σ` of `experiment` for its predicted
trajectory, concatenated over its observations. At a reset time, the incoming state
is observed.
"""
function residual(experiment::Experiment, traj)
    return reduce(vcat, map(observation -> residual(observation, traj), experiment.observations))
end

function residual(observation::Observation, traj)
    prediction = map(sample_indices(traj.t, observation.t)) do k
        observe(observation.selector, traj, k)
    end
    return weighted(observation.σ, observation.y .- prediction)
end

weighted(::Nothing, r) = r
weighted(σ, r) = r ./ σ

function observe(selector, traj, k)
    t = traj.t[k]
    return selector(traj.u[k], SymbolicIndexingInterface.parameter_values_at_time(traj, traj, t), t)
end
observe(selector::Union{Symbol, Integer}, traj, k) = traj.u[k][state_index(traj, selector)]

state_index(traj, selector) = SymbolicIndexingInterface.variable_index(traj, selector)
sample_indices(times, t) = map(ti -> findfirst(==(ti), times), t)
ChainRulesCore.@non_differentiable state_index(::Any, ::Any)
ChainRulesCore.@non_differentiable sample_indices(::Any, ::Any)

"One experiment, or a tuple or NamedTuple of experiments."
const ObservationExperiments = Union{Experiment, Tuple{Vararg{Experiment}}, NamedTuple{<:Any, <:Tuple{Vararg{Experiment}}}}

as_experiments(experiment::Experiment) = (experiment,)
as_experiments(experiments::ObservationExperiments) = experiments

"""
$(SIGNATURES)

Return the concatenated observation residuals of `experiments` for their predicted
trajectories, e.g. `map((e, s) -> first(layer(e, ps, s)), experiments, states)`.
"""
residual(experiments::Union{Tuple, NamedTuple}, trajectories::Union{Tuple, NamedTuple}) =
    reduce(vcat, values(map(residual, experiments, trajectories)))

"""
$(SIGNATURES)

Return the total number of observations of `experiments`.
"""
n_observations(experiments::Union{Tuple, NamedTuple}) = sum(n_observations, values(experiments))
