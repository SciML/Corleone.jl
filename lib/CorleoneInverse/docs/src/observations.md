# [Observations and parameters](@id inverse_observations)

## Describing measurements

An [`Experiment`](@ref) holds one [`Observation`](@ref) per measured quantity, written
as `selector => (t, y)` or `selector => (t, y, σ)`. Each quantity has its own
observation times. The selector can be

- a state symbol or index, such as `:u₁` or `1`,
- a scalar map `(u, p, t) -> y`, or
- a Julia expression such as `:(p₁ * u₁)`.

An experiment may also prescribe its initial state:

```julia
experiment = Experiment(:u₁ => (t₁, y₁, 0.05), :u₂ => (t₂, y₂, 0.05); u0 = [0.5, 2.0])
```

A shooting layer predicts an experiment with `layer(experiment, ps, st)`.
[`experiment_state`](@ref) adds the observation times to the saved grid without
moving the shooting boundaries, and [`residual`](@ref) concatenates the residuals of
all observations. With a single `Experiment`, every prediction uses the same
parameters `ps`; for values that differ between experiments, see
[multiple experiments](@ref inverse_experiments).

### Measurement expressions

A measurement expression uses the state and parameter names at the sample time;
`t` refers to that time. Expressions are compiled once, when the model is prepared:

```julia
observations = Experiment(:(p₁ * u₁ + t) => (times, values, 0.1))
```

## Estimated and fixed values

Estimated values are declared with Corleone's annotations on the shooting layer:
`Tunable(i; kind = Val(:parameter))` is estimated, `Constant(i; kind = Val(:parameter))`
is fixed, and tunable initial conditions are estimated as well.

### Transforms and priors

Annotations accept an increasing scalar `transform` from TransformVariables and a
univariate `prior` from Distributions. Both refer to natural values:

```julia
Tunable(1; kind = Val(:parameter), default = 1.2, transform = asℝ₊, prior = LogNormal(0.0, 0.5))
```

The optimizer then works in internal coordinates (here, the logarithm), while
defaults, layer outputs, and reported estimates stay in natural units. Without a
transform, bounds are applied directly to the values; with one, the optimizer
coordinates are unbounded.

## Observation parameters

Sometimes the measurement model has its own unknowns, such as a scaling gain or an
offset. Declare them with `observation_parameters`. The keyword names each value,
so the annotation only needs its default and metadata, and the names must differ
from the state, parameter, and time names:

```julia
observations = Experiment(:(u₁^2 - w + offset) => (times, values, 0.1))
model = CorleoneDynamicOptProblem(
    shooting, observations;
    observation_parameters = (
        w = Tunable(default = 0.5, lowerbound = 0.0),
        offset = Constant(default = 0.0),
    ),
)
```

The fit estimates the shooting values and the observation tunables together.
In the packed parameters, `ps.shooting` goes to the ODE and `ps.evaluation` holds the
observation tunables; constants live in `model.st.evaluation`. All experiments
share these values. A plain number, such as `w = 0.5`, is shorthand for an unbounded
tunable, and expressions see natural values when a tunable has a transform.

To evaluate the residuals of such a model by hand, save the trajectory at the
observation times first:

```julia
ps = model.ps
state = experiment_state(shooting, observations, model.st)
trajectory, _ = shooting(observations, ps.shooting, state)
residual(
    model.layer.evaluation, observations, trajectory, ps.evaluation;
    evaluation_state = model.st.evaluation,
)
```

Without `evaluation_state`, the constants take their defaults. Expressions that
read only trajectory values also work with `residual(observations, trajectory)`.
Pass `eval_module` and `evaluator` when expressions call your own helper functions.
