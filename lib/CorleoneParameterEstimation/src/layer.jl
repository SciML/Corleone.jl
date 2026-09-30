"""
    ParameterEstimationLayer{L<:Tuple, O, D} <: LuxCore.AbstractLuxWrapperLayer{:layer}

Wraps one Corleone shooting layer *per experiment* together with an observation map and the
measured data so that the parameters of the underlying differential-equation problem can be
fitted with a least-squares approach.

# Fields
$(FIELDS)

All experiments are simulated with the **same model**: every wrapped shooting layer is built
from the same differential-equation problem, and the tunable model parameters are shared
between the experiments. Each experiment is simulated from its own initial conditions (given
in the data, or the initial conditions of the problem) and sampled at its own observation
times (the shooting layer of experiment `i` is built with `saveat = data.t[i]`). All
experiments share the problem's time span `tspan`; an experiment observed over a shorter
interval only uses the model trajectory at its own observation times when building the
residuals. The least-squares residuals of all experiments are concatenated into a single
residual vector.

Like the underlying `SingleShootingLayer`/`MultipleShootingLayer`, this layer is a callable
Lux layer: [`LuxCore.setup`](@ref) produces the initial parameters and states (shared by all
experiments), and calling the layer with `(nothing, ps, st)` integrates the problem for every
experiment. A single-experiment layer returns one `Trajectory`; a multi-experiment layer
returns a tuple with one `Trajectory` per experiment. The observed quantities at the data
time points are obtained by applying the layer's `observed` map to the trajectory states.

The layer only treats the *tunable* parameters of the wrapped shooting layers as degrees of
freedom. In particular, fixing individual parameters (e.g. via `bounds_p` with equal lower
and upper bounds on the wrapped layer) is respected, so a subset of the model parameters can
be kept at known values while the rest are estimated. Tunable initial conditions
(`tunable_ic`) are shared between all experiments: the tunable entries of the state vector
are fitted jointly while the remaining entries stay at each experiment's prescribed initial
conditions.
"""
struct ParameterEstimationLayer{L<:Tuple, O, D} <: LuxCore.AbstractLuxWrapperLayer{:layer}
    "One Corleone shooting layer per experiment (all built from the same problem)"
    layer::L
    "Observation map `(u, p, t) -> y`, returning the measured quantities of a state `u` at time `t`"
    observed::O
    "The measured data"
    data::D
end

"""
$(SIGNATURES)

Construct a `ParameterEstimationLayer` from pre-built Corleone shooting layers (one per
experiment), an observation map and measured data.

# Arguments
- `layer`: A `SingleShootingLayer`/`MultipleShootingLayer`, or a tuple with one shooting
  layer per experiment. All layers must be built from the same differential-equation problem.
- `data`: An [`ExperimentData`](@ref) holding the observation time points, values and
  (optionally) initial conditions of each experiment.

# Keywords
- `observed`: An observation map `(u, p, t) -> y` extracting the measured quantities from a
  state vector `u` (and possibly parameters `p`) at time `t`. Defaults to `u -> u`, i.e. the
  full state vector is observed.

When wrapping existing shooting layers, each layer should have been built with
`saveat = data.t[i]` so the model is sampled exactly at the observation time points. The
short-cut constructor `ParameterEstimationLayer(prob, alg, data)` handles this automatically.

# Examples
```julia
prob = ODEProblem(lotka, u0, tspan, p0)
shooting = SingleShootingLayer(prob, Tsit5())
data = ExperimentData(times, y)
pe = ParameterEstimationLayer(shooting, data; observed = (u, p, t) -> u[1:2])
```
"""
function ParameterEstimationLayer(layers::Tuple, data::ExperimentData; observed = (u, p, t) -> u)
    length(layers) == n_experiments(data) || throw(ArgumentError(
        "Got $(length(layers)) shooting layer(s) for $(n_experiments(data)) experiment(s)." *
        " Provide one shooting layer per experiment."
    ))
    return ParameterEstimationLayer{typeof(layers), typeof(observed), typeof(data)}(
        layers, observed, data
    )
end

function ParameterEstimationLayer(layer, data::ExperimentData; observed = (u, p, t) -> u)
    return ParameterEstimationLayer((layer,), data; observed = observed)
end

"""
    ParameterEstimationLayer(prob, alg, data; kwargs...)

Construct a `ParameterEstimationLayer` directly from a differential-equation problem, an
integrator and data, by building one `SingleShootingLayer` per experiment from the *same*
problem. The layer of experiment `i` is built with `saveat = unique(data.t[i])` and, if the
experiment prescribes its own initial conditions, with `u0 = data.u0[i]`. All remaining
keywords are forwarded to the `SingleShootingLayer` constructors.

```julia
data = ExperimentData(
    Experiment(t1, y1; σ = 0.05, u0 = [1.0, 1.0]),
    Experiment(t2, y2; σ = 0.05, u0 = [0.4, 2.5]),
)
pe = ParameterEstimationLayer(prob, Tsit5(), data)
```
"""
function ParameterEstimationLayer(
        prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.AbstractDEAlgorithm,
        data::ExperimentData; observed = (u, p, t) -> u, kwargs...
    )
    _validate_experiments(prob, data)
    # Each experiment is simulated with the same model (same problem, shared tunable
    # parameters and tspan -- the tspan is carried by the shared layer state, so all
    # experiments must be observed within `prob.tspan`). Each shooting layer is built from
    # the problem with this experiment's initial conditions and `saveat = t_i`.
    layers = ntuple(n_experiments(data)) do i
        t_i = observation_times(data, i)
        prob_i = prob
        u0_i = initial_conditions(data, i)
        isnothing(u0_i) || (prob_i = remake(prob; u0 = u0_i))
        SingleShootingLayer(prob_i, alg; saveat = unique(t_i), kwargs...)
    end
    return ParameterEstimationLayer(layers, data; observed = observed)
end

"""
    ParameterEstimationLayer(prob, alg, data, shooting_points; kwargs...)

Construct a `ParameterEstimationLayer` on top of a `MultipleShootingLayer` built from `prob`
with the given `shooting_points`, one layer per experiment, sampling the model at the
observation time points of each experiment.
"""
function ParameterEstimationLayer(
        prob::SciMLBase.AbstractDEProblem, alg::SciMLBase.AbstractDEAlgorithm,
        data::ExperimentData, shooting_points::AbstractVector{<:Real}; observed = (u, p, t) -> u, kwargs...
    )
    _validate_experiments(prob, data)
    # Each experiment is simulated with the same model (same problem, shared tunable
    # parameters and tspan -- the tspan is carried by the shared layer state, so all
    # experiments must be observed within `prob.tspan`).
    layers = ntuple(n_experiments(data)) do i
        t_i = observation_times(data, i)
        prob_i = prob
        u0_i = initial_conditions(data, i)
        isnothing(u0_i) || (prob_i = remake(prob; u0 = u0_i))
        MultipleShootingLayer(prob_i, alg, shooting_points...; saveat = unique(t_i), kwargs...)
    end
    return ParameterEstimationLayer(layers, data; observed = observed)
end

# Validate that every experiment is compatible with the problem the layers are built from:
# prescribed initial conditions must match the state dimension, and either all experiments
# specify a noise scale σ or none (weighted vs. unweighted residuals).
function _validate_experiments(prob::SciMLBase.AbstractDEProblem, data::ExperimentData)
    n_states = length(prob.u0)
    σ_given = falses(n_experiments(data))
    for i in 1:n_experiments(data)
        u0_i = initial_conditions(data, i)
        if !isnothing(u0_i)
            length(u0_i) == n_states || throw(ArgumentError(
                "The initial conditions of experiment $i have length $(length(u0_i))," *
                " but the state of the problem has length $n_states."
            ))
        end
        σ_given[i] = !isnothing(experiments(data)[i].σ)
    end
    if any(σ_given) && !all(σ_given)
        throw(ArgumentError(
            "Either all experiments must specify a noise scale `σ` or none" *
            " (weighted vs. unweighted residuals cannot be mixed)."
        ))
    end
    return
end

function Base.show(io::IO, pe::ParameterEstimationLayer)
    type_color, no_color = SciMLBase.get_colorizers(io)
    print(
        io,
        type_color, "ParameterEstimationLayer", no_color,
        " with ", type_color, n_experiments(pe.data), no_color,
        " experiment(s) and ", type_color, n_observations(pe.data), no_color,
        " observations.\n",
    )
    print(io, no_color, "Underlying problem: ")
    return Base.show(io, "text/plain", first(pe.layer).problem)
end

"Return the tuple of shooting layers (one per experiment) wrapped by `pe`."
get_layer(pe::ParameterEstimationLayer) = pe.layer
get_observed(pe::ParameterEstimationLayer) = pe.observed
get_data(pe::ParameterEstimationLayer) = pe.data
LuxCore.initialparameters(rng::Random.AbstractRNG, pe::ParameterEstimationLayer) =
    LuxCore.initialparameters(rng, first(pe.layer))
LuxCore.initialstates(rng::Random.AbstractRNG, pe::ParameterEstimationLayer) =
    LuxCore.initialstates(rng, first(pe.layer))
LuxCore.parameterlength(pe::ParameterEstimationLayer) = LuxCore.parameterlength(first(pe.layer))

function (pe::ParameterEstimationLayer)(::Any, ps, st, args...)
    layers = pe.layer
    trajs = ntuple(n_experiments(pe.data)) do i
        first(layers[i](nothing, ps, st, args...))
    end
    return (n_experiments(pe.data) == 1 ? first(trajs) : trajs, st)
end
