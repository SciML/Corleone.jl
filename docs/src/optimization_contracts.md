# [Objectives and constraints](@id optimization_guide)

To optimize, wrap an initialized shooting layer in a
[`CorleoneDynamicOptProblem`](@ref) together with an objective and constraints, and
convert it to an Optimization.jl problem. The model evaluates the trajectory; the
solver chooses the values in `ps`.

## A complete example

This example minimizes a terminal error, subject to bounds on the state at four
times. The interior shooting node adds a continuity equality automatically.

```@example optimization_workflow
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random
using ComponentArrays, ForwardDiff, Optimization, OptimizationMOI, Ipopt

problem = ODEProblem((x, p, t) -> [p[1]], [0.0], (0.0, 2.0), [0.2])
layer = ShootingLayer(problem, Tsit5(),
    Tunable(1; kind = Val(:variable), default = 0.2,
        lowerbound = 0.0, upperbound = 2.0) => [1.0],
    Tunable(1; default = 0.2, lowerbound = 0.0, upperbound = 1.0) => [1.0])
ps, st = LuxCore.setup(MersenneTwister(42), layer)
model = CorleoneDynamicOptProblem(layer, :((u₁(2.0) - 1)^2),
    :([u₁(0.5), u₁(1.0), u₁(1.5), u₁(2.0)]);
    ps, st, lcons = zeros(4), ucons = fill(1.5, 4))
optprob = OptimizationProblem(model, AutoForwardDiff())
result = solve(optprob, Ipopt.Optimizer();
    hessian_approximation = "limited-memory", tol = 1e-8, print_level = 0)
@assert successful_retcode(result) # hide
_, reconstruct = pack(default_vectorizer(), ps)
optimized = reconstruct(result.u)
@assert maximum(abs, optimization_constraints(model, optimized, optprob.p)[5:end]) < 1e-6 # hide
(; objective = result.objective,
    constraints = optimization_constraints(model, optimized, optprob.p))
```

The first four constraint values are the sampled states; the last one is the
continuity residual, which the solver drives to zero.

When you pass `ps` and `st`, the model keeps them by reference; leave both out to
let it run setup. Building the model runs one prediction to check dimensions, and
each later objective or constraint call runs one more.

!!! note "Rebuild after structural changes"
    A prepared model, and any sparsity or AD preparation built from it, assumes fixed
    schedules, saved times, and output sizes. Rebuild it after changing any of these.
    Changing only numerical values is fine.

## [Timed expressions](@id timed_expressions)

The simplest way to write objectives and constraints is a Julia expression. A
signal followed by a time, such as `:(p₁(0.5) * u₁(1.5))`, reads that signal at that
time:

- Times must be literal numbers within the time span. Interpolate computed times,
  for example `:(u₁($(3 / 2)))`.
- The requested times are added to the integration grid, so the values are exact
  samples, not interpolations.
- At a reset, a state read returns the post-reset value; parameters are
  right-continuous.

A bare signal stands for its full saved series, so `:(sum(u₁))` sums every sample,
including both sides of resets. As an objective, a plain `Symbol` or integer selects
a terminal value. If you leave out the objective entirely, Corleone sums the
terminal values of all quadrature states.

Expressions see the functions of Corleone and Base. To call your own helpers, pass
`eval_module = @__MODULE__`; local variables are not captured, so interpolate
constants with `$`. Expressions are compiled with RuntimeGeneratedFunctions when it
is loaded, and with `eval` otherwise; see [`compile_evaluator`](@ref) to plug in
your own compiler. The [ModelingToolkit frontend](@ref modelingtoolkit_guide)
translates symbolic costs and constraints into this form.

## Callbacks

For anything an expression can't express, pass functions:

- An objective is `traj -> scalar`, or `(prediction, st) -> scalar` if it needs the
  Lux state.
- A constraint returns a scalar or a flat vector, or writes in place as
  `f!(res, traj)`.

```@example optimization_workflow
terminal(traj) = last(traj.u)[1]
function endpoints!(res, traj)
    res[1] = first(traj.u)[1]
    res[2] = last(traj.u)[1]
    return nothing
end
callback_model = CorleoneDynamicOptProblem(layer, traj -> terminal(traj)^2,
    terminal, endpoints!;
    ps, st, lcons = [0.0, [0.0, 0.0]], ucons = [2.0, [2.0, 2.0]])
length(optimization_constraints(callback_model))
```

Supply `lcons` and `ucons` with one entry per constraint, as above; for a single
vector constraint, flat vectors are enough. Constraint values are concatenated in
declaration order, followed by the shooting equalities with zero bounds. A
constraint returns a value, never a comparison: put the limits in the bounds.

`optimization_constraints(model, ps, st)` returns the constraint vector, and
`optimization_constraints!(res, model, ps, st)` writes it into a buffer. Prefer
allocating callbacks with reverse-mode AD backends that cannot differentiate
mutation. Callback models also accept other Lux containers that return several
trajectories, whereas timed expressions need a `ShootingLayer`.

## [Variable bounds](@id variable_bounds)

`lower, upper = get_bounds(layer, ps, st)` returns bound trees with the same layout
as `ps`, reading the annotation metadata without solving. Missing bounds are
infinite, and fixed values have no entries. In a larger Lux container, ordinary Lux
layers are unbounded unless they define the bound methods. The optimization bridge
flattens parameters and bounds for you. See [bounds and accuracy](@ref shooting_bounds)
for what these bounds constrain.

## Shooting equality constraints

The model appends the equalities of [multiple shooting](@ref shooting_formulation)
automatically. If you connect an optimizer yourself, get them directly:

```@example optimization_workflow
n = get_number_of_shooting_constraints(layer, ps, st) # No solve.
trajectory, next_st = layer(nothing, ps, st)
residuals = shooting_constraints(trajectory)
@assert length(residuals) == n # hide
residuals
```

Bound them by `zeros(n)` from both sides. [`shooting_constraints!`](@ref) writes
into a preallocated buffer, whose element type must accept AD numbers.

## The Optimization.jl bridge

`OptimizationProblem(model, adtype)` needs Optimization.jl and ComponentArrays to be
loaded. It flattens the variables and bounds, stores the Lux state in `optprob.p`,
and builds the objective and constraint functions. Choose `adtype` from the
[supported AD configurations](@ref automatic_differentiation).

- Pass derivative options through `function_kwargs`, for example
  `function_kwargs = (; cons_j = custom_jacobian!)`. Other keywords go to the
  optimization problem.
- Declare constraints on the model; the generated `cons` function cannot be
  replaced.
- `inplace = false` selects out-of-place constraints and derivatives.
- [Sparse derivatives](@ref sparse_derivatives) explains `sparse_constraints` and
  `sparse_hessian`.

After solving, reconstruct the parameters from `result.u` as shown below. To see
every time requested by an expression, evaluate the layer with
`(model.layer.problem, model.evaluation_grid)` as input, or call
`optimization_objective` and `optimization_constraints` with the reconstructed
parameters.

```@raw html
<span id="Sparse-constraint-Jacobians"></span>
<span id="Lagrangian-Hessian-structure"></span>
<span id="Sparse-exact-Hessians"></span>
```

## [Parameter packing](@id parameter_packing)

Optimizers work with a flat vector. `pack` creates it from the structured
parameters, together with a function that maps a vector back:

```@example optimization_workflow
q0, reconstruct = pack(default_vectorizer(), ps)
parameters = reconstruct(q0)
length(q0)
```

`reconstruct` wraps the vector it receives without copying, keeping its element
type, including AD numbers. So if you want to keep an earlier result, copy the
optimizer vector.

The default vectorizer uses ComponentArrays and requires it to be loaded. A custom
vectorizer implements [`pack`](@ref), ordering parameters and bounds identically,
and is selected with `vectorize = Val(:YourExtension)`.
