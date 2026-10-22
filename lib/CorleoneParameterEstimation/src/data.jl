"""
    Experiment(t, y; σ = nothing, u0 = nothing, controls = nothing)
    Experiment(times, ys; σ = nothing, u0 = nothing, controls = nothing)

The time-series observations of a *single experiment* of a dynamical system, together with
 the measurement noise level, the initial conditions used to simulate this experiment and
optionally the **fixed controls** applied in it.

There are two equivalent ways to describe the observations:

1. **Shared-grid matrix form** `Experiment(t, y)`: `t` is the observation time vector and
   every measured quantity is observed at *every* time point in `t`. `y` is either
   an `AbstractVector` of length `length(t)` (a single measured quantity) or an
   `AbstractMatrix` of size `(n_obs, length(t))` measuring `n_obs` quantities at each time
   point (one row per measured component, matching the ordering the layer's `observed`
   map returns).
2. **Per-quantity form** `Experiment(times, ys)`: a *different* set of observation time
   points per measured quantity. `times` and `ys` are vectors-of-vectors with one entry
   per measured quantity: the `k`-th quantity of the layer's `observed` map is observed at
   time points `times[k]` with values `ys[k]` (`length(times[k]) == length(ys[k])`). This
   is useful when e.g. different species of the model are sampled at different times, as
   chosen by an optimal experimental design.

# Arguments (matrix form)
- `t`: Observation time points as an `AbstractVector` of reals.

# Keywords
- `σ`: Measurement uncertainty (noise scale). For the shared-grid matrix form it may be
    - a scalar (shared by all observations),
    - an `AbstractVector` of length `n_obs` (per measured component), or
    - an `AbstractMatrix` of size `(n_obs, length(t))` (per observation).
    For the per-quantity form it may be a scalar (shared) or a vector-of-vectors with one
    noise series per quantity (`length(σ[k]) == length(times[k])`).
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

# Two quantities observed at different time points (per-quantity form)
exp = Experiment([prey_times, predator_times], [prey_data, predator_data]; σ = 0.05)
```
"""
struct Experiment{
    P, T<:AbstractVector, Y, S<:Union{Nothing, Number, AbstractVector, AbstractMatrix}, U, C
}
    "Observation time points (Union grid when per-quantity times are used)"
    t::T
    "Observed values: matrix `n_obs × n_t`, vector `n_t` (single quantity), or vector-of-vectors (per-quantity)"
    y::Y
    "Measurement noise scale (scalar, vector, or matrix), or `nothing`"
    σ::S
    "Initial conditions of this experiment (or `nothing` to use the problem's initial conditions)"
    u0::U
    "Fixed controls `(idx => ControlParameter)` applied in this experiment (or `nothing`)"
    controls::C
    "Per-quantity observation time vectors, or `nothing` when `y` is a shared-grid matrix"
    times::P
end

function Experiment(t::AbstractVector, y::AbstractVector; σ = nothing, u0 = nothing, controls = nothing)
    @assert length(y) == length(t) "The number of observed values must match the number of time points."
    return Experiment(t, reshape(y, :, length(t)), σ, u0, controls, nothing)
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
    return Experiment(t, y, σ, u0, controls, nothing)
end

# Per-quantity observation times: each observed quantity `k` of the layer's observation map is
# measured at its own time points `times[k]` with values `ys[k]`. The stored `t` is the (sorted)
# union grid, used e.g. to sample the model in the shooting layer; the per-quantity grid `times`
# itself distinguishes this form from the shared-grid matrix form (where `times === nothing`). Custom
# `σ` follows the same shape: a scalar (shared) or a vector-of-vectors `σ[k]` per quantity.
function Experiment(
        times::AbstractVector{<:AbstractVector}, ys::AbstractVector{<:AbstractVector};
        σ = nothing, u0 = nothing, controls = nothing
    )
    length(times) == length(ys) || throw(ArgumentError(
        "The number of per-quantity time vectors must match the number of value vectors."))
    length(times) >= 1 || throw(ArgumentError("At least one per-quantity time series is required."))
    for (tk, yk) in zip(times, ys)
        length(tk) == length(yk) || throw(ArgumentError(
            "The number of values must match the number of time points for each per-quantity series."))
    end
    t_union = sort(unique(vcat(collect.(times)...)))
    return Experiment(t_union, collect(ys), σ, u0, controls, collect(times))
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
n_observations(data::ExperimentData) = sum(_n_observed, experiments(data))

# Number of scalar observations of one experiment: `n_obs × n_t` for the shared-grid matrix
# form, or the total number of values summed over the per-quantity series.
_n_observed(e::Experiment) = isnothing(e.times) ? length(e.y) : sum(length, e.y)

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
    observation_times_per_quantity(data::ExperimentData, i::Int)

For an experiment with per-quantity observation times return the vector of per-quantity
observation time vectors; return `nothing` for an experiment using the shared-grid matrix
form. The `k`-th entry is the time vector at which the `k`-th observed quantity is measured.
"""
observation_times_per_quantity(data::ExperimentData, i::Int) = experiments(data)[i].times

"""
    per_quantity(data::ExperimentData, i::Int)

Return `true` if experiment `i` uses per-quantity observation times (i.e. different observed
quantities measured at different time points), and `false` if it uses the shared-grid matrix
form in which every quantity is measured at the same time points.
"""
per_quantity(data::ExperimentData, i::Int) = !isnothing(experiments(data)[i].times)

"""
    initial_conditions(data::ExperimentData, i::Int)

Return the initial conditions of experiment `i`, or `nothing` if the experiment does not
prescribe its own initial conditions (the initial conditions of the differential-equation
problem are used instead).
"""
initial_conditions(data::ExperimentData, i::Int) = experiments(data)[i].u0

# "1 parameter" / "2 parameters", etc. Shared by the `ExperimentData`/`Experiment` printers
# here and the `ParameterEstimate`/`ParameterEstimationResult` printers in `uncertainty.jl`.
# Pass an explicit plural for words that do not simply take an "s" (e.g. "quantity").
_count_label(n::Integer, singular::AbstractString, plural::AbstractString = singular * "s") =
    "$n $(n == 1 ? singular : plural)"

# Short description of an `Experiment`'s measurement-noise specification.
_noise_description(::Nothing) = nothing
_noise_description(σ::Number) = "σ = $σ"
_noise_description(σ::AbstractVector{<:AbstractVector}) = "σ = per-quantity"
_noise_description(σ::AbstractMatrix) = "σ = per-observation"
_noise_description(::AbstractVector) = "σ = per-component"

# One-line description of an `Experiment`, used by both printers.
function _experiment_description(e::Experiment)
    if isnothing(e.times)
        nq = e.y isa AbstractMatrix ? size(e.y, 1) : 1
        s = _count_label(nq, "quantity", "quantities") * " on a shared grid of " *
            _count_label(length(e.t), "time point")
    else
        s = _count_label(length(e.y), "quantity", "quantities") * " with per-quantity times"
    end
    isnothing(e.σ) || (s *= ", " * _noise_description(e.σ))
    isnothing(e.controls) || (s *= ", " * _count_label(length(e.controls), "fixed control"))
    isnothing(e.u0) || (s *= ", u0 = " * string(e.u0))
    return s
end

function Base.show(io::IO, e::Experiment)
    type_color, no_color = SciMLBase.get_colorizers(io)
    return print(io, type_color, "Experiment", no_color, " with ", _experiment_description(e))
end

function Base.show(io::IO, data::ExperimentData)
    type_color, no_color = SciMLBase.get_colorizers(io)
    exps = experiments(data)
    print(io, type_color, "ExperimentData", no_color, " with ")
    print(io, _count_label(length(exps), "experiment"), " and ")
    print(io, _count_label(n_observations(data), "observation"))
    n_ctrl = count(e -> !isnothing(e.controls), exps)
    n_ctrl > 0 && print(io, " (", _count_label(n_ctrl, "experiment"), " with fixed controls)")
    # For multiple experiments, list a one-line description of each (capped to avoid a
    # pathologically long print for large ensembles).
    if length(exps) > 1
        limit = 8
        for (i, e) in enumerate(exps)
            if i > limit
                print(io, "\n  … ", _count_label(length(exps) - limit, "more experiment"))
                break
            end
            print(io, "\n  [", i, "] ", _experiment_description(e))
        end
    end
    return
end
