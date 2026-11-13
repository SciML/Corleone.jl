# [Multiple experiments](@id inverse_experiments)

Real studies combine several experiments: different doses, initial conditions, or
input profiles. Usually the model parameters are shared, while each experiment
has its own controls and perhaps its own initial state. `Experiments` describes
which values are shared and which are local.

## Shared and local values

The constructor takes the problem and the shared annotations. Each named entry
lists its own annotations and observations:

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
trajectories, st = shooting(experiments, ps, st)
model = CorleoneDynamicOptProblem(shooting, experiments; ps, st)
problem = NonlinearLeastSquaresProblem(model)
```

Here parameter 1 is shared. Experiment `a` keeps the control fixed at zero, while
experiment `b` estimates its own control value and initial state.

- Ordinary model parameters are shared unless an experiment overrides them.
- Every experiment gives a `Tunable` or `Constant` annotation for each control.
  A bare control annotation switches at the union of all observation times;
  `annotation => nothing` uses one value, and `annotation => grid` sets the
  switching times explicitly.
- Local values can have their own bounds, transforms, and priors. Shared priors
  are counted once, and local priors once per experiment.
- An entry accepts `annotations`, `observations`, and `u0`. A bare `Experiment`
  is shorthand for `(observations = data,)`.

`shooting(experiments, ps, st)` runs every experiment and returns a vector of
trajectories in declaration order. `Experiments(base_shooting, specs)` builds the
container from an existing layer's annotations; the container itself stores no
solver, so the same container works with different shooting layers for the same
problem.

To look at a single experiment, call the container: `calls, st = experiments(nothing, ps, st)`
returns one merged `(experiment_ps, experiment_st)` tuple per experiment, and
`shooting(nothing, calls[i]...)` evaluates experiment `i` on its own. A fitted
model's evaluation layer runs the whole batch with the packed layout:

```julia
layer = model.layer.evaluation
trajectories, next_st = layer(experiments, model.ps, (; shooting = model.st, evaluation = (;)))
```

In the parameter tree, `ps.shared` holds the shared tunables and
`ps.experiments.<name>` the local ones; fixed values and schedules live in `st`.
`ParameterEstimate(model, q)` reports the joint covariance, including the
correlations between shared and local values, and `mean_squared_residual(model)`
averages the squared, noise-scaled residuals.

The following script builds and fits such a batch end to end:

```@example experiment_layers
using CorleoneInverse
include(joinpath(pkgdir(CorleoneInverse), "examples", "experiment_layers", "main.jl"))
```

## Reusing a fit

The solver problem stores the Lux state in `p`. When the schedules and layout stay
the same, `remake` replaces fixed values and warm-starts from a previous fit. Here
we correct a fixed input of the first experiment:

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

Adding an experiment changes the number of residuals and possibly the unknowns, so
`remake` is not enough. `append_experiment` builds a new model and solver problem,
carries over the fitted values, and initializes any new ones. The original batch
stays usable. If an inferred control grid gains times, each new interval starts
from the previous control value.

```@example reuse_fit
second_run = spec(0.25, [0.25, 0.75], [0.4375, 1.3125])
joint = append_experiment(model, updated, :second_run => second_run; u0 = fit.u)
joint_fit = solve(joint.problem, LevenbergMarquardt())
@assert joint.problem.u0 ≈ fit.u # hide
@assert joint_fit.u ≈ [1.5] # hide
joint_fit.u
```

## Constraints across experiments

Constraints are passed after the experiments. A pair `tag => expression` evaluates
an expression on one experiment, and a callback receives all trajectories, so it
can couple experiments. Bounds use the usual `lcons` and `ucons`, and shooting
equalities are appended automatically. Constrained fits need `OptimizationProblem`:

```julia
model = CorleoneDynamicOptProblem(
    shooting, experiments,
    :a => :(u₁(2.0)),
    trajectories -> last(trajectories.a.u)[1] - last(trajectories.b.u)[1];
    ps, st, lcons = [0.0, 0.0], ucons = [10.0, 0.0],
)
opt = OptimizationProblem(model, AutoForwardDiff())
```

Experiments in a tuple are tagged by integer index. With a single experiment, the
tag can be left out. Times in timed expressions are added to that experiment's
saved grid when the model is prepared.

## Sparse derivatives

Experiments are only coupled through their shared values, so the residual
Jacobian is mostly empty. In this example, two experiments share the rate `p₁`, the
observation parameter `w`, and a fixed offset, and each one fits its own initial
state. The constraint reads only experiment `a` at time `0.75`:

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

`residual_jacobian_sparsity` returns one row per measurement. The first two
residuals depend on the initial state of `a`, and the last two on that of `b`; the
fixed offset has no column. The constraint touches columns 1 and 2, and the Hessian
has no entry between the two initial states, `H[2, 3]`.

The default reducers add curvature one residual at a time, which keeps these zeros.
A custom reducer may couple residuals and therefore gets a denser pattern.

To use the patterns, load DifferentiationInterface and SparseMatrixColorings:

```julia
using Optimization
import DifferentiationInterface, SparseMatrixColorings

opt = OptimizationProblem(
    model, AutoForwardDiff(); sparse_constraints = true, sparse_hessian = true,
)
```

Rebuild the model after changing the experiments, schedules, or observation times.
