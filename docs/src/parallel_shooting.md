# Parallel shooting

`ShootingLayer(...; ensemblealg = EnsembleThreads())` solves independent shooting
blocks using the existing `mythreadmap` dispatch. `EnsembleDistributed()` uses
`Distributed.pmap`; workers must load Corleone, the solver, and any user-defined
model functions. `EnsembleSerial()` remains the default.

A new block starts at an interior time where **every state variable** has a
scheduled initial condition. Both tunable and constant assignments count. Partial
resets and control changes stay inside a block and run sequentially, using the
existing solution buffers. A reset at the final time does not create a block.
With no independent interior restart, the layer uses the serial solve directly.
The segment grid is computed during setup and stored in `st.timestops`. Changing
schedules does not require reconstructing the layer, but requires explicitly
refreshing that cache before the next evaluation:

```julia
st = merge(st, (; timestops = Corleone.get_timestops(layer, nothing, st)))
```

Forward calls retain this vector in the returned state. The independent block
partition is still derived from the cached grid and current reset schedules on
each call.

Internally, `shooting_states(layer, ps, st, tstops)` prepares an ordered vector of
states with sliced `timestops` and private scheduling buffers. Adjacent slices
share their boundary timepoint and retain every control switch and partial reset
inside the interval. The ensemble map iterates over these states and calls the
existing serial solver with the same `ps` for each one. Numerical values in state
retain their references and differentiation paths through this preparation.

Each independent block solves from its assigned start values. Once those solves
finish, the layer computes each block's first residual against the preceding
block's actual endpoint. This retains the dependence of the continuity loss on
both blocks. Fixed assignments contribute no residuals, matching serial behavior.
The trajectory uses the solutions' actual saved samples, including `saveat`
samples, and preserves reset boundaries in chronological order.

Each block owns its activity vector and lookup caches. Within a process, numerical
parameters and values held in Lux state retain their references; they are not
detached from the differentiation graph. Distributed workers receive serialized
inputs instead. The returned state is the last block's state. Cache contents
can differ from serial execution because that block visits only its own times.
Treat grids and parameter arrays as read-only during an evaluation. Model
functions and callbacks must also be safe for concurrent calls; shared mutable
callback state is not isolated by the layer. Concurrent *outer* calls need their
own Lux state, including when a call falls back to serial execution.

## Differentiation

ForwardDiff follows the parallel forward path with the same shared numerical
inputs. Regression tests compare threaded and serial gradients, including
continuity residuals, parameters shared across blocks, and numerical values stored
in Lux state, at two inputs and against finite differences.

For ChainRules reverse AD, `shooting_map` has a configured rule that delegates to
ordinary `map` during differentiation. This records and solves blocks serially
and uses the backend's map pullback to accumulate shared-input cotangents. It
does not repeat the forward solves. Parallel reverse execution is deferred until
task-local AD contexts and safe cotangent reduction are implemented. Block-result
assembly has a dedicated pullback so Zygote receives consistent tuple cotangents
from the solution, residual, and returned-state paths.

The maintained `Differentiation` test group checks full-solver ForwardDiff and
Zygote gradients for serial and threaded layers against an analytic reference.
The `Distributed` group checks ForwardDiff through two local workers against
serial gradients and finite differences. For distributed ForwardDiff, define a
named objective function or callable type in a module loaded on every worker.
Serializing an anonymous objective closure can change its type on a worker,
causing a ForwardDiff tag-ordering error during nested solver differentiation.
The regression fixture loads the same objective and model definitions on every
process before computing gradients.

ReverseDiff, Mooncake, and Enzyme have not been validated through threaded or
distributed execution. See [Automatic differentiation](automatic_differentiation.md)
for the serial solver's additional backend requirements.

## Current limits

Parallel dispatch currently accepts ODE problems. A full state reset does not
necessarily reset a DAE's derivative state, so DAEs need a separate independence
criterion. Random processes and other problem types also need explicit handling.
All blocks must complete their requested time spans; early termination callbacks
are not supported as a way to cancel later blocks.

Parallel execution allocates private block buffers and then concatenates their
outer vectors; saved numerical state arrays are retained by reference. For cheap
ODEs or very short blocks, scheduling and assembly overhead can outweigh any
speedup. Benchmark the actual model, reset density, and thread/worker count.

`test/core/shooting_parallel.jl` and `test/core/shooting_parallel_ad.jl` run in the
Core group. Run the worker checks with
`GROUP=Distributed julia --project=. -e 'using Pkg; Pkg.test()'`; the group starts
two local workers and removes them afterward. Run the full-solver backend checks
with `GROUP=Differentiation julia --project=. -e 'using Pkg; Pkg.test()'`.
See `test/README.md` for the test groups and their scope. `benchmark/shooting_parallel.jl` compares warmed
serial/threaded calls for cheap and expensive right-hand sides in an environment
containing BenchmarkTools. Use `--threads=4` (or the desired thread count).

### Initial forward measurements

On 24 October 2026, Julia 1.12.7 with four threads, the benchmark above measured
the following medians over 500 warmed calls per case (`evals=1`). Both cases use
one state, 120 integration segments and 12 independent blocks. The expensive
right-hand side adds a sum of 256 sine terms. Compilation and construction are
excluded; no AD backend was loaded in the benchmark process.

| Workload | Serial | Four threads | Speedup |
|---|---:|---:|---:|
| Cheap right-hand side | 276.6 µs | 153.2 µs | 1.81× |
| Expensive right-hand side | 6.576 ms | 1.808 ms | 3.64× |

Both workloads allocated 638,216 bytes in 12,102 allocations serially and 687,408
bytes in 12,615 allocations with threads. These are illustrative workloads, not
a performance guarantee for a different model or reset schedule.
