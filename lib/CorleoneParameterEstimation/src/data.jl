"""
    Experiment(t, y; σ = nothing, u0 = nothing, controls = nothing)

The time-series observations of a *single experiment* of a dynamical system, together with
the measurement noise level, the initial conditions used to simulate this experiment and
optionally the **fixed controls** applied in it.

# Arguments
- `t`: Observation time points as an `AbstractVector` of reals.
- `y`: Observed values. Either
    - an `AbstractVector` of length `length(t)` for a single measured quantity, or
    - an `AbstractMatrix` of size `(n_obs, length(t))` measuring `n_obs` quantities
      at each time point (one row per measured component, matching the ordering the
      layer's `observed` map returns).

# Keywords
- `σ`: Measurement uncertainty (noise scale). May be
    - a scalar (shared by all observations),
    - an `AbstractVector` of length `n_obs` (per measured component), or
    - an `AbstractMatrix` of size `(n_obs, length(t))` (per observation).
    When provided, weighted residuals `(y_obs - y_model) ./ σ` are used in the
    least-squares problem. When `nothing`, unweighted residuals are used.
- `u0`: Initial conditions used to simulate this experiment. Either `nothing` (the initial
  conditions of the differential-equation problem the layer is built from are used) or an
  `AbstractVector` of the same length as the state vector of the problem.
- `controls`: A tuple of `control_index => ControlParameter` pairs (the same form as the
  `controls` keyword of `SingleShootingLayer`), specifying the *fixed* piecewise-constant
  control functions applied in this experiment. Control values are read from the
  `controls` field of each `ControlParameter` and are **not** estimated: they are treated as
  fixed inputs of the experiment design (e.g. produced by optimal experimental design).

# Examples
```julia
t = collect(0.0:0.5:5.0)
y = [0.1, 0.25, 0.6, 1.1, 1.9, 3.0, 4.5, 6.2, 8.1, 10.2, 12.5]
exp = Experiment(t, y; σ = 0.05, u0 = [1.0])

# An experiment with a fixed control acting on parameter 1
exp = Experiment(t, y; σ = 0.05, u0 = [1.0],
    controls = (1 => ControlParameter(collect(0.0:1.0:5.0), controls = ones(5)),))
```
"""
struct Experiment{
    T<:AbstractVector, Y, S<:Union{Nothing, Number, AbstractVector, AbstractMatrix}, U, C
}
    "Observation time points"
    t::T
    "Observed values (matrix `n_obs × n_t` or vector `n_t` for a single component)"
    y::Y
    "Measurement noise scale (scalar, vector, or matrix), or `nothing`"
    σ::S
    "Initial conditions of this experiment (or `nothing` to use the problem's initial conditions)"
    u0::U
    "Fixed controls `(idx => ControlParameter)` applied in this experiment (or `nothing`)"
    controls::C
end

function Experiment(t::AbstractVector, y::AbstractVector; σ = nothing, u0 = nothing, controls = nothing)
    @assert length(y) == length(t) "The number of observed values must match the number of time points."
    return Experiment(t, reshape(y, :, length(t)), σ, u0, controls)
end

function Experiment(t::AbstractVector, y::AbstractMatrix; σ = nothing, u0 = nothing, controls = nothing)
    @assert size(y, 2) == length(t) "The number of columns of `y` must match the number of time points."
    if !isnothing(σ)
        if σ isa AbstractVector
            @assert length(σ) == size(y, 1) "Per-component `σ` must match the number of observed components."
        elseif σ isa AbstractMatrix
            @assert size(σ) == size(y) "`σ` must match the size of `y` when given as a matrix."
        end
    end
    return Experiment(t, y, σ, u0, controls)
end

"""
    ExperimentData(t, y; σ = nothing, u0 = nothing)
    ExperimentData(experiments::Experiment...)
    ExperimentData(experiments::AbstractVector{<:Experiment})

Container for the time-series observations of one or more *experiments* of a dynamical
system that are used to fit model parameters.

A single experiment can be constructed directly from its observation times and values.
Multiple experiments are constructed from one or more [`Experiment`](@ref) objects. All
experiments are simulated with the **same model** and share the same (tunable) parameters;
each experiment may have its own observation times, measurement noise and initial
conditions. The least-squares residuals of all experiments are concatenated into a single
residual vector.

# Arguments (single-experiment constructor)
- `t`: Observation time points as an `AbstractVector` of reals.
- `y`: Observed values, see [`Experiment`](@ref).

# Keywords (single-experiment constructor)
- `σ`: Measurement uncertainty, see [`Experiment`](@ref).
- `u0`: Initial conditions used to simulate this experiment, see [`Experiment`](@ref).

# Examples
```julia
# A single experiment
t = collect(0.0:0.5:5.0)
y = [0.1, 0.25, 0.6, 1.1, 1.9, 3.0, 4.5, 6.2, 8.1, 10.2, 12.5]
data = ExperimentData(t, y; σ = 0.05)

# Two experiments with different initial conditions, simulated with the same model
data = ExperimentData(
    Experiment(t, y; σ = 0.05, u0 = [1.0]),
    Experiment(t2, y2; σ = 0.05, u0 = [2.0]),
)

# Existing single-experiment `ExperimentData`s can also be combined directly:
data1 = ExperimentData(t, y; σ = 0.05, u0 = [1.0])
data2 = ExperimentData(t2, y2; σ = 0.05, u0 = [2.0])
data = ExperimentData(data1, data2)
```
"""
struct ExperimentData{N, E<:NTuple{N, Experiment}}
    "One [`Experiment`](@ref) per dataset"
    experiments::E

    function ExperimentData(exps::NTuple{N, Experiment}) where {N}
        N >= 1 || throw(ArgumentError("ExperimentData requires at least one experiment."))
        return new{N, typeof(exps)}(exps)
    end
end

ExperimentData(exps::Experiment...) = ExperimentData(exps)
ExperimentData(exps::AbstractVector{<:Experiment}) = ExperimentData(Tuple(exps))

# Disambiguate `ExperimentData(::Experiment...)` from `ExperimentData(::ExperimentData...)`
# for the empty call, which is invalid (at least one experiment is required).
ExperimentData() = throw(ArgumentError("ExperimentData requires at least one experiment."))

"""
    ExperimentData(datas::ExperimentData...)

Combine the experiments of multiple [`ExperimentData`](@ref) containers into a single
`ExperimentData`. Each `datas[i]` may itself hold one or more experiments; all of them are
flattened into the combined container and their residuals are concatenated when fitting.
"""
function ExperimentData(datas::ExperimentData...)
    return ExperimentData(Tuple(Iterators.flatten(experiments(d) for d in datas)))
end

function ExperimentData(t::AbstractVector, y::AbstractVector; σ = nothing, u0 = nothing, controls = nothing)
    return ExperimentData((Experiment(t, y; σ = σ, u0 = u0, controls = controls),))
end

function ExperimentData(t::AbstractVector, y::AbstractMatrix; σ = nothing, u0 = nothing, controls = nothing)
    return ExperimentData((Experiment(t, y; σ = σ, u0 = u0, controls = controls),))
end

"""
    controls(data::ExperimentData, i::Int)

Return the fixed controls (a tuple of `control_index => ControlParameter` pairs) of experiment
`i`, or `nothing` if that experiment does not prescribe fixed controls.
"""
controls(data::ExperimentData, i::Int) = experiments(data)[i].controls

"""
    has_controls(data::ExperimentData)

Return `true` if **any** experiment of `data` prescribes fixed controls.
"""
has_controls(data::ExperimentData) = any(e -> !isnothing(e.controls), experiments(data))

"""
    experiments(data::ExperimentData)

Return the tuple of [`Experiment`](@ref)s held by `data`.
"""
experiments(data::ExperimentData) = data.experiments

"""
    n_experiments(data::ExperimentData)

Return the number of experiments held by `data`.
"""
n_experiments(data::ExperimentData) = length(data.experiments)

"""
    n_observations(data::ExperimentData)

Return the total number of scalar observations `(n_obs × n_t)`, summed over all experiments.
"""
n_observations(data::ExperimentData) = sum(length(e.y) for e in experiments(data))

"""
    observation_times(data::ExperimentData)
    observation_times(data::ExperimentData, i::Int)

Return the observation time points of `data`. For a single-experiment `data` the time points
of that experiment are returned; for multiple experiments a tuple with the time points of
each experiment is returned. The two-argument variant returns the time points of experiment
`i`.
"""
function observation_times(data::ExperimentData)
    return n_experiments(data) == 1 ? first(experiments(data)).t :
        map(e -> e.t, experiments(data))
end
observation_times(data::ExperimentData, i::Int) = experiments(data)[i].t

"""
    initial_conditions(data::ExperimentData, i::Int)

Return the initial conditions of experiment `i`, or `nothing` if the experiment does not
prescribe its own initial conditions (the initial conditions of the differential-equation
problem are used instead).
"""
initial_conditions(data::ExperimentData, i::Int) = experiments(data)[i].u0
