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
Lux layer: `LuxCore.setup` produces the initial parameters and states (shared by all
experiments), and calling the layer with `(nothing, ps, st)` integrates the problem for every
experiment. A single-experiment layer returns one `Trajectory`; a multi-experiment layer
returns a tuple with one `Trajectory` per experiment. The observed quantities at the data
time points are obtained by applying the layer's `observed` map to the trajectory states.

The layer only treats the *tunable* parameters of the wrapped shooting layers as degrees of
freedom. In particular, fixing individual parameters (e.g. via `bounds_p` with equal lower
and upper bounds on the wrapped layer) is respected, so a subset of the model parameters can
be kept at known values while the rest are estimated. To estimate only a *chosen subset* of
the model parameters (and keep every other parameter fixed at its value in `prob.p`), pass
the `params` keyword to the convenience constructors
`ParameterEstimationLayer(prob, alg, data; params = ...)`; `params` is forwarded to the
underlying `SingleShootingLayer`/`MultipleShootingLayer`. Tunable initial conditions
(`tunable_ic`) are shared between all experiments: the tunable entries of the state vector
are fitted jointly while the remaining entries stay at each experiment's prescribed initial
conditions.
"""
struct ParameterEstimationLayer{L <: Tuple, O, D} <: LuxCore.AbstractLuxWrapperLayer{:layer}
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
    length(layers) == n_experiments(data) || throw(
        ArgumentError(
            "Got $(length(layers)) shooting layer(s) for $(n_experiments(data)) experiment(s)." *
                " Provide one shooting layer per experiment."
        )
    )
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
    # Each experiment is simulated with its own `SingleShootingLayer` (and thus its own
    # layer state) built from the same problem, so each experiment may use its own fixed
    # controls (grid and values) independently. Experiments that do not specify controls are
    # simulated with the control-slot parameters held constant at their default value from
    # `prob.p`. The tunable model parameters (all indices that are not control slots) and
    # tunable initial conditions are shared optimization variables across experiments.
    layers = ntuple(n_experiments(data)) do i
        t_i = observation_times(data, i)
        prob_i = prob
        u0_i = initial_conditions(data, i)
        isnothing(u0_i) || (prob_i = remake(prob; u0 = u0_i))
        c_i = _experiment_layer_controls(data, i, prob)
        SingleShootingLayer(prob_i, alg; saveat = unique(t_i), controls = c_i, kwargs...)
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
    # Each experiment has its own `MultipleShootingLayer` (and layer state); controls are
    # provided per experiment, and experiments without controls keep the control-slot
    # parameters constant at their `prob.p` default.
    layers = ntuple(n_experiments(data)) do i
        t_i = observation_times(data, i)
        prob_i = prob
        u0_i = initial_conditions(data, i)
        isnothing(u0_i) || (prob_i = remake(prob; u0 = u0_i))
        c_i = _experiment_layer_controls(data, i, prob)
        MultipleShootingLayer(prob_i, alg, shooting_points...; saveat = unique(t_i), controls = c_i, kwargs...)
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
            length(u0_i) == n_states || throw(
                ArgumentError(
                    "The initial conditions of experiment $i have length $(length(u0_i))," *
                        " but the state of the problem has length $n_states."
                )
            )
        end
        σ_given[i] = !isnothing(experiments(data)[i].σ)
    end
    if any(σ_given) && !all(σ_given)
        throw(
            ArgumentError(
                "Either all experiments must specify a noise scale `σ` or none" *
                    " (weighted vs. unweighted residuals cannot be mixed)."
            )
        )
    end
    _validate_controls(data)
    return
end

# Validate fixed-control specifications. The control *indices* are a property of the
# model (controls replace parameters), so every experiment that specifies controls must
# control the same parameter indices; experiments without controls are simulated with those
# control-slot parameters held constant at their `prob.p` default. Each experiment may use
# its own control grid and values (it gets its own shooting layer and layer state).
function _validate_controls(data::ExperimentData)
    has_controls(data) || return nothing
    # Every control entry must be a `control_index => ControlParameter` pair.
    for i in 1:n_experiments(data)
        c_i = controls(data, i)
        isnothing(c_i) && continue
        for pair in c_i
            pair isa Pair || throw(
                ArgumentError(
                    "Fixed controls must be `control_index => ControlParameter` pairs;" *
                        " got an entry of type $(typeof(pair)) in experiment $i."
                )
            )
            last(pair) isa Corleone.ControlParameter || throw(
                ArgumentError(
                    "Fixed controls must be `control_index => ControlParameter` pairs;" *
                        " got a value of type $(typeof(last(pair))) in experiment $i."
                )
            )
        end
    end
    # All experiments with controls must control the same parameter indices.
    structure = _shared_control_structure(data)
    for i in 1:n_experiments(data)
        c_i = controls(data, i)
        isnothing(c_i) && continue
        _assert_matching_control_indices(i, c_i, structure)
    end
    return
end

# Return the tuple of `control_index => ControlParameter` pairs defining the model-level
# control slots, taken from the first experiment that specifies controls, or `nothing` if no
# experiment does. The discretization grid of this reference is only used as the default for
# experiments that specify no controls; controlled experiments keep their own grid.
function _shared_control_structure(data::ExperimentData)
    for i in 1:n_experiments(data)
        c_i = controls(data, i)
        isnothing(c_i) || return c_i
    end
    return nothing
end

function _assert_matching_control_indices(i, c_i, structure)
    length(c_i) == length(structure) || throw(
        ArgumentError(
            "Experiment $i specifies $(length(c_i)) fixed control(s) but the model has" *
                " $(length(structure)) control slot(s); all experiments must control the same" *
                " parameter indices."
        )
    )
    for (idx, _) in structure
        any(q -> first(q) == idx, c_i) || throw(
            ArgumentError(
                "Experiment $i must specify fixed controls for shared control index $idx."
            )
        )
    end
    return
end

# The `controls` tuple used to build the shooting layer of experiment `i`.
#
# - If experiment `i` specifies controls, its own `(index => ControlParameter)` pairs (grid
#   and fixed values) are used.
# - If experiment `i` specifies no controls but the model has control slots, every control
#   slot is held at its *default* value from `prob.p`, as a constant piecewise control on the
#   reference control grid.
# - If no experiment specifies controls, `nothing` is returned (no controls at all).
function _experiment_layer_controls(data::ExperimentData, i::Int, prob)
    structure = _shared_control_structure(data)
    isnothing(structure) && return ()
    c_i = controls(data, i)
    isnothing(c_i) && return _experiment_layer_controls_uncontrolled(structure, prob)
    # A valued experiment keeps its own grid and values; only the shared control slots are
    # used (every experiment must control the same parameter indices).
    tspan = prob.tspan
    return ntuple(length(structure)) do k
        idx, ref_cp = structure[k]
        me = findfirst(q -> first(q) == idx, c_i)
        isnothing(me) && throw(
            ArgumentError(
                "Experiment $i must specify fixed controls for shared control index $idx."
            )
        )
        cp = last(c_i[me])
        Corleone.control_length(cp; tspan = tspan) == Corleone.control_length(ref_cp; tspan = tspan) ||
            throw(
            ArgumentError(
                "The control discretization of experiment $i on index $idx does not match" *
                    " the reference control grid (control slots must have a common" *
                    " discretization across experiments)."
            )
        )
        idx => cp
    end
end

# The `controls` tuple for an experiment that does *not* specify controls but whose model has
# control slots: each slot held constant at its default value from `prob.p` on the reference
# grid. (Used when an experiment supplies `controls = nothing` yet other experiments control
# the same parameter indices.)
function _experiment_layer_controls_uncontrolled(structure, prob)
    p_full, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.p)
    tspan = prob.tspan
    return ntuple(length(structure)) do k
        (idx, ref_cp) = structure[k]
        nlen = Corleone.control_length(ref_cp; tspan = tspan)
        values = fill(p_full[idx], nlen)
        idx => Corleone.ControlParameter(ref_cp.t; name = ref_cp.name, controls = values, bounds = ref_cp.bounds)
    end
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
    return Base.show(io, "text/plain", Corleone.get_problem(first(pe.layer)))
end

"Return the tuple of shooting layers (one per experiment) wrapped by `pe`."
get_layer(pe::ParameterEstimationLayer) = pe.layer
get_observed(pe::ParameterEstimationLayer) = pe.observed
get_data(pe::ParameterEstimationLayer) = pe.data
# The shared layer parameters. For a single-shooting layer (or a multiple-shooting layer
# with a single experiment) the shared `(u0, p, controls)` / interval parameters are
# returned. For multiple shooting with several experiments every experiment has its own
# local shooting states, so a tuple with one parameter set per experiment is returned; this
# matches the tuple accepted by the layer callable (`_experiment_ps_ms`).
function LuxCore.initialparameters(rng::Random.AbstractRNG, pe::ParameterEstimationLayer)
    if _is_multiple_shooting(pe) && n_experiments(pe.data) > 1
        return map(l -> LuxCore.initialparameters(rng, l), pe.layer)
    end
    return LuxCore.initialparameters(rng, first(pe.layer))
end

function LuxCore.parameterlength(pe::ParameterEstimationLayer)
    if _is_multiple_shooting(pe) && n_experiments(pe.data) > 1
        return sum(l -> LuxCore.parameterlength(l), pe.layer)
    end
    return LuxCore.parameterlength(first(pe.layer))
end

# Each experiment gets its *own* shooting-layer state, built from its own layer, so that
# experiments may use different (or no) controls and grids independently.
function LuxCore.initialstates(rng::Random.AbstractRNG, pe::ParameterEstimationLayer)
    return map(l -> LuxCore.initialstates(rng, l), pe.layer)
end

function (pe::ParameterEstimationLayer)(::Any, ps, st::Tuple, args...)
    layers = pe.layer
    data = pe.data
    trajs = ntuple(n_experiments(data)) do i
        ps_i = _experiment_ps(pe, layers[i], ps, i)
        first(layers[i](nothing, ps_i, st[i], args...))
    end
    return (n_experiments(data) == 1 ? first(trajs) : trajs, st)
end

# Build the parameter NamedTuple used to simulate experiment `i`, dispatching on the kind
# of shooting layer wrapped by `pe`.
#
# * **Single shooting**: `ps` is the shared `(u0, p, controls)` NamedTuple; the fixed
#   control values of experiment `i` (from the data, or the parameter default for
#   controlless experiments) are merged in. When the model has no controls at all the
#   shared `ps` is returned unchanged (its control vector is empty). The control values
#   are baked into each layer and are never part of the fitted optimization variables.
# * **Multiple shooting**: `ps` is either a *tuple* with one `(interval_1, ...)` NamedTuple
#   per experiment (produced by `_unflatten`), in which case every experiment already
#   carries its own local shooting states and fixed controls and `ps[i]` is used directly;
#   or a single shared interval NamedTuple (one experiment, or a caller reusing one `ps`
#   for all experiments), in which case the shared local states are kept and only the
#   fixed controls of experiment `i` are swapped in.
function _experiment_ps(pe::ParameterEstimationLayer, layer, ps, i::Int)
    if layer isa Corleone.MultipleShootingLayer
        return _experiment_ps_ms(pe, ps, i)
    end
    data = pe.data
    has_controls(data) || return ps
    ctrl = LuxCore.initialparameters(Random.default_rng(), pe.layer[i]).controls
    return merge(ps, (controls = ctrl,))
end

function _experiment_ps_ms(pe::ParameterEstimationLayer, ps::Tuple, i::Int)
    return ps[i]
end

function _experiment_ps_ms(pe::ParameterEstimationLayer, ps::NamedTuple, i::Int)
    ctrl = LuxCore.initialparameters(Random.default_rng(), pe.layer[i])
    blocks = ntuple(
        k -> begin
            cb = ctrl[k]
            merge(ps[k], (controls = cb.controls,))
        end, length(ps)
    )
    return NamedTuple{keys(ps)}(blocks)
end
