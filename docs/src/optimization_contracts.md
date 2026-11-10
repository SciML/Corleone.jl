# [Adding objectives and constraints](@id Optimization-facing-contracts)

Wrap an initialized shooting layer in [`CorleoneDynamicOptProblem`](@ref), then
construct an Optimization.jl problem. The model evaluates trajectory objectives
and constraints; the solver chooses the parameters.

## Prepared optimization models

This example minimizes a terminal error subject to four sampled state limits.
The interior tunable state introduces a continuity equality automatically.

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

Supplying `ps` and `st` together retains them by reference; omitting both runs
setup once. Preparation performs one prediction to validate dimensions and saved
indices. Each later objective or constraint evaluation makes one prediction;
there is no shared numerical-solution cache. Reprepare after changing schedules,
saved layout, expression times, or callback output sizes.

## Callbacks and structured bounds

An objective returns a real scalar: `traj -> scalar`, or `(prediction, st) -> scalar`
when it needs the current optimizer state. The one-argument form takes precedence.
A constraint returns a real scalar or flat vector, or writes into `f!(res, traj)`.
In-place return values are ignored. Bounds specify the output sizes:

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

Supply both `lcons` and `ucons`, with one matching top-level subtree per constraint.
A single vector constraint accepts flat bound vectors directly. Constraint outputs
are flattened in declaration order, followed by automatic shooting equalities
with zero bounds. Comparisons are not numerical constraints: return a value and
supply its limits separately.

`optimization_constraints(model, ps, st)` allocates the output;
`optimization_constraints!(res, model, ps, st)` writes into a buffer of the exact
length. In-place callbacks receive their own slices. Use allocating callbacks for
reverse AD backends that cannot differentiate mutation. Callback-based Lux
containers may also return collections of trajectories; timed expressions and
terminal selectors require a `ShootingLayer`.

## Timed Julia expressions

An expression such as `:(p₁(0.5) * u₁(1.5))` reads signals at specified times.
Times must be finite real literals within the problem span; interpolate computed
times, for example `:(u₁($(3 / 2)))`. Preparation adds requested times to the
integration grid and resolves indices against actual saved samples, including
`saveat` samples. It performs no interpolation. At resets, state reads select the
post-reset sample; parameter reads are right-continuous.

Within an expression, a bare signal denotes its full saved series:
`:(sum(u₁))` includes both sides of resets. A plain Symbol or integer objective
selector instead denotes a terminal value. Omitting the objective sums terminal
quadratures and requires quadrature metadata. The
[ModelingToolkit frontend](modelingtoolkit.md) translates symbolic costs and
constraints into this interface.

Expressions resolve functions and constants in `eval_module` (Corleone by default,
including Base functions). Use `eval_module = @__MODULE__` for global helpers or
interpolate literal constants; expressions do not capture local bindings.
`default_evaluator()` selects RuntimeGeneratedFunctions when loaded, otherwise
Julia eval. See [`compile_evaluator`](@ref) for custom evaluator implementations.

## Variable bounds

`lower, upper = get_bounds(layer, ps, st)` returns trees with the parameter layout
and array shapes of `ps`. Fixed values in Lux state have no optimizer entries.
Missing bounds are infinite. The helper reads metadata without setup or solving.
A larger Lux container can include shooting layers and ordinary layers, whose
leaves are unbounded unless they specialize the bound methods.

Flatten parameters and both bound trees with the same vectorizer. The optimization
bridge does this automatically. State bounds apply to stored initial/shooting
values; use sampled constraints for limits elsewhere. See
[numerical accuracy](shooting.md#Bounds-and-numerical-accuracy) for the limits of
sampled feasibility checks.

## Shooting equality constraints

The model appends the equalities defined by the [shooting formulation](@ref shooting_formulation).
For a custom optimizer integration, use:

```@example optimization_workflow
n = get_number_of_shooting_constraints(layer, ps, st) # No solve.
trajectory, next_st = layer(nothing, ps, st)
residuals = shooting_constraints(trajectory)
@assert length(residuals) == n # hide
residuals
```

Set both equality bounds to `zeros(n)`. Use [`shooting_constraints!`](@ref) to
write into a preallocated buffer; its element type must accommodate AD values.
State residuals precede inserted-parameter residuals; their signs and ordering
are defined in [shooting](shooting.md). The helpers add neither an objective nor
sampled constraints. Run [`add_parameter_segments!`](@ref), if needed, **before**
packing parameters, querying bounds, or preparing a model.

## SciMLBase optimization bridge

Load Optimization.jl, a solver, and ComponentArrays for the default bridge.
`OptimizationProblem(model, adtype)` packs variables and bounds, stores Lux state
in `optprob.p`, and creates objective and constraint callbacks. With no constraints,
`cons`, `lcons`, and `ucons` are `nothing`. Choose a supported
[differentiation configuration](automatic_differentiation.md).

Pass derivative options through `function_kwargs`, for example
`function_kwargs = (; cons_j = custom_jacobian!)`; other keywords configure the
optimization problem. Declare constraints on the model: the bridge's generated
`cons` callback cannot be replaced. `inplace = false` selects allocating constraints
and the out-of-place derivative convention. See [sparse derivatives](sparse_derivatives.md)
for optional Jacobian and Hessian preparation.

After solving, reconstruct parameters from `result.u`. For timed expressions,
evaluate with `(model.layer.problem, model.evaluation_grid)` as the layer input
to include all requested times, or use `optimization_objective` and
`optimization_constraints` with the reconstructed parameters and current state.

## [Sparse derivatives](@id Sparse-constraint-Jacobians)

```@raw html
<span id="Lagrangian-Hessian-structure"></span>
<span id="Sparse-exact-Hessians"></span>
```

See [Preparing sparse derivatives](sparse_derivatives.md) for colored constraint
Jacobians, Lagrangian structure, and exact Hessians.

## Parameter packing interface

```@example optimization_workflow
q0, reconstruct = pack(default_vectorizer(), ps)
parameters = reconstruct(q0)
length(q0)
```

ComponentArrays reconstruction wraps the **current** vector with the retained
axes and scalar type, including AD values. Packing a native NamedTuple allocates
flat storage; packing an existing ComponentVector reuses its backing storage.
Retain or copy optimizer vectors deliberately when preserving earlier results.

Custom vectorizers implement [`pack`](@ref), returning a flat vector and a callable
that restores named parameter access from its argument. They must order parameters
and bounds identically. Select one with `vectorize = Val(:YourExtension)`;
`default_vectorizer()` requires a supported loaded extension, currently ComponentArrays.
