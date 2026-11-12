"""
$(TYPEDEF)

Measurements `y` of one observed quantity at the sorted, unique times `t`. The
`selector` is a state symbol or index, a map `(u, p, t) -> y` returning a scalar,
or a scalar Julia expression such as `:(p₁ * u₁)`. Expressions use state and
parameter values at each observation time. Prepared fits compile them once.
The noise scale `σ` is `nothing` (unweighted residuals), a scalar, or a vector
matching `y`.

# Fields
$(FIELDS)
"""
@concrete terse struct Observation
    "The state selector, scalar observation callback, or Julia observable expression."
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

n_observations(experiment::Experiment) = sum(observation -> length(observation.y), experiment.observations)

function experiment_sample_times(problem::SciMLBase.AbstractSciMLProblem, experiment::Experiment)
    t0, tf = problem.tspan
    all(t -> t0 <= t <= tf, experiment.t) ||
        throw(ArgumentError("Observation times must lie within the problem time span."))
    isnothing(experiment.u0) || length(experiment.u0) == length(problem.u0) ||
        throw(DimensionMismatch("An experiment's u0 must match the problem's state length."))
    return experiment.t
end

function get_saveat(problem::SciMLBase.AbstractSciMLProblem, samples, schedules, existing = nothing)
    times = isnothing(existing) ? copy(samples) : vcat(samples, existing)
    configured = get(problem.kwargs, :saveat, ())
    if configured isa Number
        configured = reduce(vcat, map(stops -> interval_saveat(configured, times, stops), schedules))
    end
    isempty(configured) || (times = vcat(times, collect(configured)))
    return sort!(unique!(times))
end

# A scalar solver saveat is relative to the start of each shooting segment.
function interval_saveat(interval, samples, tstops)
    step = abs(interval)
    ranges = map(1:(length(tstops) - 1)) do i
        lo, hi = sampling_tspan((tstops[i], tstops[i + 1]), samples)
        (lo + step):step:hi
    end
    return reduce(vcat, ranges)
end

get_saveat(layer::ShootingLayer, experiment::Experiment, st) =
    get_saveat(layer, (experiment,), st)

function get_saveat(layer::ShootingLayer, experiments, st)
    times = reduce(vcat, map(experiment -> experiment_sample_times(layer.problem, experiment), values(experiments)))
    return get_saveat(layer.problem, times, (st.timestops,), get(st, :saveat, nothing))
end

"""
$(SIGNATURES)

Return the Lux state `st` of `layer` with observation times merged with existing
and solver-configured sampling times in `saveat`. The shooting boundaries in
`st.timestops` are preserved.
"""
function experiment_state(layer::ShootingLayer, experiment::Experiment, st)
    return merge(st, (; saveat = get_saveat(layer, experiment, st)))
end

"""
$(SIGNATURES)

Predict `experiment` from its initial state on the shooting grid in `st`, saving
the observation times prepared by [`experiment_state`](@ref).
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
is observed. For expressions, `eval_module` supplies custom functions and
constants and `evaluator` selects the compilation strategy.
"""
function residual(experiment::Experiment, traj; eval_module::Module = @__MODULE__, evaluator = default_evaluator())
    return residual_from_samples(nothing, experiment, traj, (), eval_module, evaluator)
end

function residual(observation::Observation, traj; eval_module::Module = @__MODULE__, evaluator = default_evaluator())
    return residual_from_samples(nothing, observation, traj, (), eval_module, evaluator)
end

function residual_from_samples(layer, experiment::Experiment, traj, values, eval_module, evaluator)
    return reduce(vcat, map(experiment.observations) do observation
        residual_from_samples(layer, observation, traj, values, eval_module, evaluator)
    end)
end

function residual_from_samples(layer, observation::Observation, traj, values, eval_module, evaluator)
    indices = sample_indices(traj.t, observation.t)
    all(!isnothing, indices) || throw(ArgumentError("Every observation time must be saved in its trajectory."))
    selector = observation.selector
    if selector isa Expr
        selector = compile_observation_expression(selector, SymbolicIndexingInterface.symbolic_container(traj), traj.t, layer, eval_module, evaluator)
    end
    return observation_residual(observation, traj, indices, values, selector)
end

function observation_residual(observation, traj, indices, values = (), selector = observation.selector)
    prediction = map(k -> observe(selector, traj, k, values), indices)
    return weighted(observation.σ, observation.y .- prediction)
end

weighted(::Nothing, r) = r
weighted(σ, r) = r ./ σ

observe(selector, traj, k, values) = observe(selector, traj, k)
function observe(selector, traj, k)
    t = traj.t[k]
    return selector(traj.u[k], SymbolicIndexingInterface.parameter_values_at_time(traj, traj, t), t)
end
observe(selector::Union{Symbol, Integer}, traj, k) = traj.u[k][state_index(traj, selector)]

state_index(traj, selector) = SymbolicIndexingInterface.variable_index(traj, selector)
sample_indices(times, t) = map(ti -> findfirst(==(ti), times), t)
ChainRulesCore.@non_differentiable state_index(::Any, ::Any)
ChainRulesCore.@non_differentiable sample_indices(::Any, ::Any)

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
n_observations(experiments::Union{Tuple, NamedTuple})::Int =
    sum(n_observations, values(experiments))
