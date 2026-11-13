# [Parallel shooting](@id parallel_shooting)

With multiple shooting, segments that start from assigned states don't depend on
each other, so they can be solved at the same time. Select the execution with the
`ensemblealg` keyword of `ShootingLayer`:

| `ensemblealg` | Execution |
|---|---|
| `EnsembleSerial()` | One segment after another (default). |
| `EnsembleThreads()` | Threads; start Julia with, for example, `julia --threads=4`. |
| `EnsembleDistributed()` | Worker processes. |

```@example parallel_shooting
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random

problem = ODEProblem((x, p, t) -> [p[1] * x[1]], [1.0], (0.0, 3.0), [-1.0])
layer = ShootingLayer(problem, Tsit5(),
    Tunable(1; kind = Val(:variable), default = [0.4, 0.2]) => [1.0, 2.0],
    Constant(1) => nothing; ensemblealg = EnsembleThreads())
ps, st = LuxCore.setup(MersenneTwister(42), layer)
trajectory, next_st = layer(nothing, ps, st)
shooting_constraints(trajectory)
```

The result is the same trajectory, with the same [reset samples](@ref trajectory_guide)
and [residuals](@ref shooting_formulation), as with serial execution.

## Which segments run in parallel

An interior reset starts an independent block only if it assigns **every state**,
whether tunable or fixed. Partial resets and parameter switches stay inside a
block, and resets at the final time create no block. Without any independent block,
the layer solves serially. Parameters remain shared between blocks unless you split
them with [`add_parameter_segments!`](@ref).

Parallel execution supports ODE problems. A full reset does not make a DAE's
algebraic states or a random process independent, so these are not split.
All blocks must reach the end of their spans; a callback that terminates one block
early cannot cancel the others.

## Thread safety and distributed workers

Each block works on its own copies of the activity and lookup caches. Beyond that:

- Treat grids and numerical buffers as read-only during a call, and make model
  functions and callbacks safe to call concurrently.
- Concurrent calls of the same layer need separate Lux states, even if execution
  falls back to serial solving.
- The returned Lux state is that of the last block, so its caches can differ from a
  serial run; see [changing a schedule](@ref schedule_changes) to refresh them.

For distributed execution, start the workers with the intended project and load
Corleone, the solver, and the model functions on every worker. Workers receive
serialized copies of the inputs. **ForwardDiff** additionally needs the objective to
be a named function or callable type defined on every worker: anonymous functions
can change type during serialization and cause tag-ordering errors.

## Differentiation and performance

ForwardDiff runs through the parallel forward solve. Reverse-mode AD records the
blocks serially and adds up their contributions, so the reverse pass is not
parallel; see [automatic differentiation](@ref automatic_differentiation).

Parallel execution has overhead for buffers, scheduling, and assembly, and cheap or
short segments may not benefit. Measure your own model with
`benchmark/shooting_parallel.jl`; dated measurements are in the
[validation archive](automatic_differentiation_history.md#Initial-forward-measurements).
