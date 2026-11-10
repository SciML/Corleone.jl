# [Running parallel shooting](@id Parallel-shooting)

Use `ensemblealg = EnsembleThreads()` for threaded ODE block solves, or
`EnsembleDistributed()` for worker processes. `EnsembleSerial()` is the default.
Start Julia with the required thread count, for example `julia --threads=4`.

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

## Independent blocks

An interior reset starts a new block only when **every state** has an assigned
value there. Both tunable and fixed assignments count. Partial resets and parameter
switches stay within sequential blocks; final-time resets create no block. Without
an independent interior restart, evaluation uses the serial solve.

After solving, each block's first matching residual is computed against the
preceding block's actual endpoint. The [residual conventions](shooting.md) and
[reset samples](trajectories.md) are the same as for serial shooting. Parameters
remain shared unless explicitly split with [`add_parameter_segments!`](@ref).

## State and worker ownership

Each block has private activity and lookup caches. Within a process, numerical
parameters and fixed values retain their references and AD paths. Distributed
workers receive serialized inputs. The returned Lux state is the last block's
state; its caches can differ from serial execution. See
[scheduling](scheduling.md#Defaults-and-schedule-changes) to refresh cached grids.

Treat grids and numerical buffers as read-only during evaluation. Model functions
and callbacks must be safe for concurrent calls; shared mutable callback state is
not isolated. Concurrent outer layer calls need separate Lux state, including
when execution falls back to serial solving.

For distributed execution, start workers with the intended project and load
Corleone, the solver, and named model functions on every worker. Distributed
ForwardDiff also requires the same named objective function or callable type on
every worker; serialized anonymous objectives can cause tag-ordering errors.

## Differentiation and limits

ForwardDiff follows the parallel forward path. ChainRules reverse AD records block
solves serially and accumulates shared-input cotangents; it does not provide
parallel reverse execution. See the [AD guide](automatic_differentiation.md) for
maintained backend configurations and fixed-grid restrictions.

Parallel dispatch supports ODE problems. A full state reset alone does not establish
independence for a DAE's derivative state or a random process. All blocks must
complete their requested spans; early termination callbacks cannot cancel later blocks.

Private buffers, scheduling, and assembly add overhead. Measure the actual model,
reset density, and worker/thread count with `benchmark/shooting_parallel.jl`;
cheap or short blocks may not benefit. Dated measurements are in the
[validation archive](automatic_differentiation_history.md#Initial-forward-measurements).
