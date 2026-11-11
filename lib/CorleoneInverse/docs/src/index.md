# [Parameter estimation](@id inverse_index)

CorleoneInverse fits the tunable parameters of a Corleone `ShootingLayer` to
measurements. Corleone provides the experiment and observation types. An [`Experiment`](@ref) stores one [`Observation`](@ref) per observed
quantity as a `selector => (t, y)` or `selector => (t, y, σ)` pair, where the
selector is a state symbol or index, a scalar map `(u, p, t) -> y`, or a Julia
expression such as `:(p₁ * u₁)`. Symbolic maps compile during problem preparation. Each quantity
has its own observation times; an experiment may also prescribe its initial state:

```julia
experiment = Experiment(:u₁ => (t₁, y₁, 0.05), :u₂ => (t₂, y₂, 0.05); u0 = [0.5, 2.0])
```

A shooting layer predicts an experiment with `layer(experiment, ps, st)`, where
[`experiment_state`](@ref) merges the observation times into the segment grid of
`st`. In this observation-only interface, experiments share the layer's parameters
`ps`; use callable experiment layers below for independent local values. [`residual`](@ref)
concatenates the observation residuals

```math
r = \frac{y - h(u(t), p, t)}{\sigma}
```

over all observations and experiments.

## Parameters, transforms, and priors

Tunable parameters, fixed parameters and bounds are declared with Corleone's
annotations on the shooting layer: `Tunable(i; kind = Val(:parameter))` is estimated,
`Constant(i; kind = Val(:parameter))` is fixed. Tunable initial conditions are
estimated as well. Annotations also accept an increasing scalar `transform` from
TransformVariables and a univariate `prior` from Distributions, both on natural
values:

```julia
Tunable(1; kind = Val(:parameter), default = 1.2, transform = asℝ₊, prior = LogNormal(0.0, 0.5))
```

The Lux parameters, and hence the optimizer, then work in internal coordinates,
while layer outputs and defaults stay natural. Without an explicit transform,
`BoundedTransform` stores identity-coordinate bounds. Explicit TransformVariables
transforms replace it and use unbounded optimizer coordinates.

## Solving

Prepare one [`CorleoneDynamicOptProblem`](@ref) from the shooting solver and
experiments. Measurements define the objective, with shared/local priors counted
once per owned value. The model retains initialized `ps` and `st`.

```julia
model = CorleoneDynamicOptProblem(shooting, experiments; ps, st)
fit = solve(NonlinearLeastSquaresProblem(model), LevenbergMarquardt())
estimate = ParameterEstimate(model, fit.u)
```

`NonlinearLeastSquaresProblem(model)` uses the prepared observation residuals.
Untransformed Normal priors add Gaussian residual rows. Use
`OptimizationProblem(model, ad)` for annotation bounds, general priors, and
constraints. The scalar objective is half the weighted residual sum of squares
minus `log_prior`. The convenience builders accepting `(shooting, experiments,
ps, st)` prepare this same model internally.

### Symbolic measurements and constraints

Each measurement expression uses current-time state and parameter names. The
independent variable refers to its sample time. The observation supplies the
sampling grid and noise scale, for example

```julia
observations = Experiment(:(p₁ * u₁ + t) => (times, values, 0.1))
```

Tagged constraints select an experiment, and timed signal calls request their
sampling points during preparation. Callbacks receive the collection of
trajectories and can couple experiments. Ordinary `lcons`/`ucons` bounds apply,
and shooting equalities are appended automatically.

```julia
model = CorleoneDynamicOptProblem(
    shooting, experiments,
    :a => :(u₁(2.0)),
    trajectories -> last(trajectories.a.u)[1] - last(trajectories.b.u)[1];
    ps, st, lcons = [0.0, 0.0], ucons = [10.0, 0.0],
)
opt = OptimizationProblem(model, AutoForwardDiff())
```

For observation tuples, integer tags select entries. A single experiment accepts
an untagged constraint expression. Tagged callbacks can use allocating or in-place
constraint forms. Observable maps and sample indices compile once. Reprepare the
model after changing selectors, observation times, or schedules.

## Shared and local values with Experiments

`Experiments` is a data-only Lux container. It defines shared and local model
parameters, controls, and initial variables, without storing a solver or shooting
layer. Its constructor takes the problem and shared annotations; each named entry
can override any parameter or variable locally:

```julia
experiments = Experiments(problem,
    Tunable(1) => nothing;
    experiments = (
        a = (observations = data_a,),
        b = (
            annotations = (
                Tunable(1; default = 1.2) => nothing, # Local ordinary parameter
                Tunable(1; kind = Val(:variable), default = 2.0) => [0.0],
            ),
            observations = data_b,
        ),
    ),
)
shooting = ShootingLayer(problem, alg)
ps, st = LuxCore.setup(rng, experiments)
calls, st = experiments(nothing, ps, st)
trajectories, st = shooting(experiments, ps, st)
model = CorleoneDynamicOptProblem(shooting, experiments; ps, st)
problem = NonlinearLeastSquaresProblem(model)
```

`calls` is a tuple of merged `(experiment_ps, experiment_st)` tuples. Calling the
container prepares values only. `ShootingLayer` dispatch executes all experiments
and returns a **vector of trajectories** in declaration order, together with the
updated experiment state. Named definitions retain their names in `ps` and `st`.
Individual tuples can also be evaluated with `shooting(nothing, calls[i]...)`;
the state carries the prepared local annotation configuration. A different
shooting solver can reuse the same container for the same problem.

Preparation and execution use generated traversal of the named fields in Lux state,
with literal accesses for each experiment and parameter branch. This preserves
inference for native parameters and ComponentArray views, including mixed local
grids, constants, and transforms. Output remains a vector: matching result types
use a concrete element type, and mixed result types use a finite union of concrete
trajectory types. State field order may differ without changing declaration order.


`Experiments(base_shooting, specs)` may inherit an existing layer's definitions at
construction; the resulting container retains no shooting layer or solver.

Base definitions remain shared unless locally overridden. Local `Tunable` and
`Constant` annotations use the same `annotation => timegrid` syntax as
`ShootingLayer`, including independent grids, transforms, bounds, and priors.
Ordinary model parameters may be local as well as controls and initial variables.
Global definitions overridden everywhere are omitted from setup, excluding unused
unknowns, priors, and initializer calls.

Each specification accepts `annotations`, `observations`, and `u0`. Observations
accept an observation-only `Experiment` or a tuple of observation pairs. An
`Experiment` alone is shorthand for `(observations = data,)`; `(;)` has no local
definitions or observations. `Experiments` is the sole exported container type for
this feature; its internal local-value layers are private.

`ps.shared.initialcondition` and `.parameter` contain global tunables;
`ps.experiments.<name>.initialcondition` and `.parameter` contain local tunables.
Fixed values and schedules live in `st`. ComponentArrays are supported. Local
constant overrides replace the corresponding numerical and state branches.
Prepared configurations retain the metadata needed for distinct grids and
transforms; no annotation processing, initialization, or problem remaking happens
in prediction callbacks. Shared priors are counted once, local priors once each.

Use `OptimizationProblem(model, ad)` for bounds, general priors, and multiple
shooting. `ParameterEstimate(model, q)` reports joint covariance, including
shared/local cross terms.

```@example experiment_layers
using CorleoneInverse
include(joinpath(pkgdir(CorleoneInverse), "examples", "experiment_layers", "main.jl"))
```

## Statistics

[`ParameterEstimate`](@ref)`(model, sol.u)` evaluates the observation Jacobian `J` at the fit and
returns the covariance `(JᵀJ / s² + H)⁻¹`, where `H` is the Hessian of the negative
log prior, restricted to the null space of equality constraints, including
multiple-shooting matches. This approximation assumes an interior optimum relative
to inequalities and parameter bounds. Known noise scales use `s² = 1` and normal
quantiles; otherwise `s²` is the residual variance and intervals use Student-t
quantiles. Results are reported in natural coordinates: the covariance with the delta
method, the confidence intervals by mapping the internal endpoints.

## Worked examples

The tutorial gallery includes examples adapted from Christoph Plate
([@chplate](https://github.com/chplate))'s
[PR #168](https://github.com/SciML/Corleone.jl/pull/168).

- [Lotka–Volterra estimation](@ref inverse_lotka_plain)
- [Joint Lotka–Volterra experiments](@ref inverse_lotka_multiexperiment)
- [OED-designed fixed controls](@ref inverse_lotka_oed_control)
- [Multiple shooting with continuity constraints](@ref inverse_multiple_shooting)
- [SIR estimation across two regions](@ref inverse_sir_multicity)
- [Robertson DAE estimation](@ref inverse_rober_dae)
- [Lux universal differential equation](@ref inverse_ude_growth)
