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
[`experiment_state`](@ref) adds the observation times to `st.saveat` and preserves
the shooting boundaries in `st.timestops`. In this observation-only interface,
experiments share the layer's parameters `ps`; use callable experiment layers
below for independent local values. [`residual`](@ref)
concatenates the observation residuals

```math
r = \frac{y - h(u(t), p, t)}{\sigma}
```

over all observations and experiments. A fit must provide a noise scale for
every observation group or omit it for every group. With no scales, residuals
are unweighted.

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

```@example first_fit
using CorleoneInverse, OrdinaryDiffEqTsit5, NonlinearSolve

ode = ODEProblem((u, p, t) -> -p[1] .* u, [1.0], (0.0, 2.0), [0.5])
shooting = ShootingLayer(ode, Tsit5(), Tunable(1) => nothing)
t = collect(0.2:0.2:2.0)
experiment = Experiment(1 => (t, exp.(-1.3 .* t), 0.01))
model = CorleoneDynamicOptProblem(shooting, experiment)
fit = solve(NonlinearLeastSquaresProblem(model), LevenbergMarquardt())
ParameterEstimate(model, fit.u)
```

The [`CorleoneDynamicOptProblem`](@ref) initializes and retains `ps` and `st`.
Pass both explicitly to reuse an existing Lux setup.

| Solver problem | Supported features |
| --- | --- |
| `NonlinearLeastSquaresProblem(model)` | Prepared observation residuals, annotation bounds, and Gaussian residual rows for untransformed Normal priors. |
| `OptimizationProblem(model, ad)` | Annotation bounds, general priors, user constraints, and multiple-shooting equalities. |

By default the scalar objective is the mean squared observation
residual minus `log_prior`, with each observed value weighted equally after noise
scaling. Pass `reducer = x -> sum(x) / 2` to the model
constructor for the Gaussian half-sum objective. The reducer receives the
squared, noise-scaled observation residuals. Priors and constraints are handled
separately. The convenience builders accepting `(shooting, experiments, ps, st)`
prepare this same model internally.

### Symbolic measurements and constraints

Each measurement expression uses current-time state and parameter names. The
independent variable refers to its sample time. The observation supplies the
sampling grid and noise scale, for example

```julia
observations = Experiment(:(p₁ * u₁ + t) => (times, values, 0.1))
```

An observation expression can also use shared trainables that belong to the
observation model. Declare them with `observation_parameters`. The keyword
names each value, so the annotation only needs its default and metadata.
The names must differ from the state, ODE parameter, and time names.

```julia
observations = Experiment(:(u₁^2 - w + offset) => (times, values, 0.1))
model = CorleoneDynamicOptProblem(
    shooting, observations;
    observation_parameters = (
        w = Tunable(default = 0.5, lowerbound = 0.0),
        offset = Constant(default = 0.0),
    ),
)
ps = model.ps
# Save the trajectory at the observation times before evaluating residuals.
state = experiment_state(shooting, observations, model.st)
trajectory, _ = shooting(observations, ps.shooting, state)
residual(
    model.layer.evaluation, observations, trajectory, ps.evaluation;
    evaluation_state = model.st.evaluation,
)
```

The fit packs the shooting values and observation tunables together. The ODE
receives `ps.shooting`, and all experiments share the tunables in `ps.evaluation`
and constants in `model.st.evaluation`. A numeric declaration such as
`w = 0.5` is shorthand for an unbounded tunable. Expressions see natural
values when a tunable has a transform. Direct evaluation uses the constant
defaults when `evaluation_state` is omitted. Expressions that read only
trajectory values also work with `residual(observations, trajectory)`. The direct residual methods accept
`eval_module` and `evaluator` when expressions use custom helpers.

#### Experiment sparsity

For an ODE fit, `residual_jacobian_sparsity(model)` returns one row per
measurement in residual order. Shared parameters use the same packed columns
in every experiment, and local parameters keep their own columns. The existing
`constraint_jacobian_sparsity(model)` and `lagrangian_hessian_sparsity(model)`
queries use the same packing. Constraint rows include shooting equalities.

In this example, two experiments share the ODE rate `p₁`, the observation
parameter `w`, and a fixed offset. Each experiment fits its own initial state.
The tagged constraint reads only experiment `a` at time `0.75`.

```@example experiment_sparsity
using CorleoneInverse, ComponentArrays, OrdinaryDiffEqTsit5

ode = ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 1.0), [2.0])
shooting = ShootingLayer(ode, Tsit5(), Tunable(1) => nothing)
times = [0.25, 0.75]

specification(x0, measurements) = (;
    annotations = Tunable(1; kind = Val(:variable), default = x0) => [0.0],
    observations = Experiment(:(u₁^2 - w + offset) => (times, measurements, 0.5)),
)
batch = Experiments(shooting, (
    a = specification(1.0, [1.75, 5.75]),
    b = specification(2.0, [5.75, 11.75]),
))
model = CorleoneDynamicOptProblem(
    shooting, batch, :a => :(u₁(0.75));
    observation_parameters = (w = Tunable(default = 0.5, lowerbound = 0.0), offset = Constant(default = 0.0)),
    lcons = 0.0, ucons = 3.0,
)

J = residual_jacobian_sparsity(model)
C = constraint_jacobian_sparsity(model)
H = lagrangian_hessian_sparsity(model)

# Columns are (p₁, initial state of a, initial state of b, w).
Int.(Matrix(J))
```

The first two residuals depend on the initial state of `a`, and the last two
depend on the initial state of `b`. The fixed offset has no column.
The tagged constraint has support in
columns 1 and 2. The Hessian entry between the two initial states, `H[2, 3]`,
is absent.

The default mean and the sum of squared residuals contribute curvature one
residual at a time. This preserves zeros between parameters from different
experiments when their only connection is a shared parameter. A custom reducer
may couple residuals, so it uses their combined dependency set. Priors and
constraints contribute their own curvature.

Load DifferentiationInterface and SparseMatrixColorings and pass
`sparse_constraints = true` or `sparse_hessian = true` to
`OptimizationProblem(model, ad)` to prepare colored derivatives.

```julia
using Optimization
import DifferentiationInterface, SparseMatrixColorings

opt = OptimizationProblem(
    model, AutoForwardDiff(); sparse_constraints = true, sparse_hessian = true,
)
```

The residual pattern is available separately for inspection. Rebuild after changing the
packing, schedules, saved samples, or observation layout.

#### Tagged constraints

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

`Experiments` is a data-only Lux container for shared model parameters,
experiment-local controls, and shared or local initial variables. Its constructor
takes the problem and shared annotations. Each named entry supplies its controls
and can define local initial variables.

```julia
experiments = Experiments(problem,
    Tunable(1) => nothing;
    controls = [2],
    experiments = (
        a = (annotations = Constant(2; default = 0.0) => nothing, observations = data_a),
        b = (
            annotations = (
                Tunable(2; default = 1.2) => nothing,
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


An `EvaluationLayer` can call the same batch directly. For a fit with observation
parameters, use the fit's packed-tree layout and the original experiment Lux state.

```julia
layer = model.layer.evaluation
trajectories, next_st = layer(experiments, model.ps, (; shooting = model.st, evaluation = (;)))
```

`Experiments(base_shooting, specs)` may inherit an existing layer's definitions at
construction; the resulting container retains no shooting layer or solver.

Ordinary model parameters are shared. Each experiment supplies a `Tunable` or
`Constant` annotation for every control. A bare control annotation uses the
batch-wide union of observation times as its switching grid. The first value
applies from the start of the problem, including when the first observation
comes later. Use `annotation => nothing` for one value throughout the
experiment, or `annotation => timegrid` to specify switching times. Local
controls and initial variables can have their own transforms, bounds, and priors.

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

Use `OptimizationProblem(model, ad)` for general priors and multiple
shooting. `ParameterEstimate(model, q)` reports joint covariance, including
shared/local cross terms. `mean_squared_residual(model)` averages the squared,
noise-scaled observation residuals, with each observed value counted once.

### Reuse a fit and add an experiment

The solver problem stores the Lux state in `p`. Use `remake` to replace fixed
values when the schedules and layouts match the prepared model. Here Lux setup
creates replacement state with a corrected fixed input and the same experiment
name and control schedule.

```@example reuse_fit
using CorleoneInverse, LuxCore, NonlinearSolve, OrdinaryDiffEqTsit5, Random

ode = ODEProblem((u, p, t) -> [p[1] + p[2]], [0.0], (0.0, 1.0), [1.0, 0.0])
shooting = ShootingLayer(ode, Tsit5(), Tunable(1) => nothing; controls = [2])
spec(control, t, y) = (
    annotations = Constant(2; default = control) => nothing,
    observations = Experiment(1 => (t, y)),
)
first_run = spec(0.0, [0.5, 1.0], [1.0, 2.0])
batch = Experiments(shooting, (; first_run))
model = CorleoneDynamicOptProblem(shooting, batch)
problem = NonlinearLeastSquaresProblem(model)
fit = solve(problem, LevenbergMarquardt())
@assert fit.u ≈ [2.0] # hide

corrected_batch = Experiments(shooting, (first_run = spec(0.5, [0.5, 1.0], [1.0, 2.0]),))
_, state = LuxCore.setup(Xoshiro(1), corrected_batch)
updated = remake(problem; u0 = fit.u, p = state)
fit = solve(updated, LevenbergMarquardt())
@assert fit.u ≈ [1.5] # hide
fit.u
```

Appending an experiment changes the residual layout and can change the sampling
grid and the unknowns. `append_experiment` prepares a new model and solver problem
from the updated state, carries over the fitted unknowns, and initializes any new
ones. The original batch remains usable. If an inferred control grid gains times,
each new interval starts from the previous control value at that time.

```@example reuse_fit
second_run = spec(0.25, [0.25, 0.75], [0.4375, 1.3125])
joint = append_experiment(model, updated, :second_run => second_run; u0 = fit.u)
joint_fit = solve(joint.problem, LevenbergMarquardt())
@assert joint.problem.u0 ≈ fit.u # hide
@assert joint_fit.u ≈ [1.5] # hide
joint_fit.u
```

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

With priors, the covariance assumes known noise scales and a fit that minimizes
half the weighted residual sum of squares minus the log prior. Use least squares
for supported Normal priors or `reducer = x -> sum(x) / 2` with
`OptimizationProblem`. The default mean-squared objective gives the prior a
different relative weight.

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

The [EGF signaling example](@ref inverse_egf_signaling) fits a published
signaling model to six stimulation doses with shared observation gains and
checks which parameter combinations the data identify.
