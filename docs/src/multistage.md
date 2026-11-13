# [Composing multi-stage problems](@id multistage_guide)

Some problems change their model partway through the horizon: a reactor switches
to a second vessel, or a new species enters an ecosystem. [`Stages`](@ref) chains
several [`ShootingLayer`](@ref)s, one per stage, each with its own
`ODEProblem`, states, and controls. A connector between two stages decides how the
next stage starts.

| Connector | How the next stage starts |
|---|---|
| [`StageTransition`](@ref) | From a function of the previous trajectory (substitution). |
| [`StageLinkage`](@ref) | From a free interface state, tied to the previous stage by a constraint. |

## A two-stage example

The first stage drives ``\dot{x} = v_1`` from ``0``. The second stage keeps
``x`` and adds a running cost ``\dot{q} = x^2``. The transition copies ``x`` and
starts the cost at zero. We want ``x`` close to ``1`` at the end of stage 1, a small
cost ``q`` in stage 2, and ``x \ge 0.5`` at the very end.

```@example multistage
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random
using ComponentArrays, ForwardDiff, Optimization, OptimizationMOI, Ipopt

stage1 = ShootingLayer(
    ODEProblem((x, p, t) -> [p[1]], [0.0], (0.0, 1.0), [0.5]), Tsit5(),
    Tunable(1; lowerbound = -2.0, upperbound = 2.0) => nothing; controls = [1])
stage2 = ShootingLayer(
    ODEProblem((x, p, t) -> [p[1], x[1]^2], [0.0, 0.0], (0.0, 1.0), [0.5]), Tsit5(),
    Tunable(1; lowerbound = -2.0, upperbound = 2.0) => nothing;
    controls = [1], quadratures = [2])
transition = StageTransition((trajectory, _) -> [last(trajectory.u)[1], 0.0])
stages = Stages(stage1, transition, stage2)

optprob = OptimizationProblem(stages, AutoForwardDiff(), Val(:ComponentArrays);
    loss = ((:((u₁ - 1)^2), 1), (:u₂, 2)),
    constraints = ((:u₁, 2) => (t = [1.0], bounds = (0.5, Inf)),))
result = solve(optprob, Ipopt.Optimizer(); tol = 1e-12, print_level = 0)

ps, st = LuxCore.setup(MersenneTwister(1), stages)
optimized = result.u + zero(ComponentArray(ps))
@assert isapprox(only(optimized.stage_1.parameter.parameters.p₁.parameter.ps), 11 / 16; atol = 1e-5) # hide
@assert isapprox(only(optimized.stage_3.parameter.parameters.p₁.parameter.ps), -3 / 16; atol = 1e-5) # hide
final, _ = stages(nothing, optimized, st)
(; objective = result.objective, final_state = last(final.u))
```

The two stages pull in different directions: stage 1 wants ``x`` at ``1``, but
stage 2 inherits that value and pays for it in ``q``. The optimum compromises at
``v_1 = 11/16``, and the terminal constraint is active, with
``v_2 = 0.5 - v_1 = -3/16``.
Calling `stages(nothing, ps, st)` runs every stage in order and returns the final
stage's trajectory.

## Objectives and constraints

The keywords of `OptimizationProblem(stages, adtype, vectorizer; ...)` are passed
to `CorleoneDynamicOptProblem(stages, loss, constraints...)`:

- `loss` is a single term or a tuple of terms that are summed. A term is a bare
  `Symbol` or expression, evaluated at the end of the **last** stage, or a pair
  `(term, k)` evaluated at the end of shooting stage `k`.
- `constraints` is a tuple of `(expression, k) => (t = times, bounds = (lb, ub))`
  pairs. The expression is evaluated at each listed time of stage `k`, in that
  stage's own time span. Leaving out `k` selects the last stage.
- `parameter_constraints = f => (lb = ..., ub = ...)` constrains a function
  `f(ps)` of the whole parameter tree, for example the sum of all stage durations.

!!! note "Two ways to count stages"
    The stage index `k` counts **shooting layers** only. The parameter and state
    trees count every entry, connectors included. In the example above, `stage2`
    is shooting stage `2` in objectives and constraints, but its values live
    under `ps.stage_3`.

Each stage's own multiple-shooting continuity equalities are added automatically,
followed by one equality block per `StageLinkage`. Bounds come from each stage's
annotations and from the connectors' own parameters.

## Transitions and linkages

`StageTransition(f; p, bounds)` calls `f(trajectory, p)` and uses the result as the
next stage's initial state. The optional parameters `p` are optimized with the
stages, so a transition can carry a tunable jump or switching cost.

`StageLinkage` instead makes the next initial state a free variable ``u_0`` and
adds the constraint

```math
\Delta_{\min} \le u_0 - m(x_f, \theta) \le \Delta_{\max},
```

where ``x_f`` is the previous stage's final state. The map ``m`` is the identity
for `StageLinkage(u0)`, affine for `StageLinkage(A, B, c)`, or any callable
`StageLinkage(m; u0)`. The default bounds are zero, which gives an exact link. A
linkage keeps each stage's solve independent of the previous one during
optimization, much like a multiple-shooting node between stages.

## Free stage durations

A stage's time span is fixed. To optimize its length, write the stage on the
normalized time ``\tau \in [0, 1]``, scale the right-hand side by a duration
parameter ``T``, and make ``T`` tunable. Use `parameter_constraints` to bound the
sum of the durations. The [switching-time tutorial](@ref lotka_sto) and the
[minimal-time tutorial](@ref lotka_min_time) use this technique.

## Tutorials

- [Two-stage chemical reactor](@ref two_stage_reactor): free stage durations and a
  constraint on the second stage.
- [Multistage Lotka–Volterra](@ref lotka_multistage): a third species enters at
  the stage switch.
- [Switching-time optimization](@ref lotka_sto): bang-bang control through stage
  durations.
