# Automatic differentiation

AD compatibility depends on both the outer backend and the sensitivity strategy
used by the SciML solve. A passing forward solve does not establish that a
particular gradient configuration works.

## Maintained coverage

The test suite checks the following configurations through full ODE solves,
including saved-state and continuity losses. See the [validation sources](#Validation-sources)
for reproduction instructions.

| Backend | Serial ODE solves | Threaded layer | Distributed layer |
|---|---|---|---|
| ForwardDiff | Tested | Tested | Tested with two local workers and a named objective |
| ReverseDiff | Tested with full specialization and pass-through sensitivities | Not established | Not established |
| Zygote | Tested with the default sensitivity path | Tested; reverse block evaluation runs serially | Not established |

These are the maintained configurations, not guarantees for arbitrary solver
algorithms, user callbacks, or array types. Compiled ReverseDiff tapes, GPU solves,
DAEs, SDEs, higher-order derivatives, and differentiation of schedule times are
outside this coverage.

Historical serial experiments also passed with Mooncake reverse mode and recorded
Enzyme forward and default ReverseDiff failures. Their exact settings, versions,
and measurements are preserved in the [validation archive](automatic_differentiation_history.md).
They do not extend the maintained matrix above.

## Choosing a configuration

### ForwardDiff

ForwardDiff works through the regular layer call with the tested default solver
configuration. Reconstruct structured parameters from the **current** optimizer
vector inside the differentiated function:

```@example forwarddiff
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, ComponentArrays, ForwardDiff

problem = ODEProblem((x, p, t) -> [p[1] * x[1]], [1.0], (0.0, 2.0), [-1.0];
    abstol = 1e-10, reltol = 1e-10)
layer = ShootingLayer(problem, Tsit5())
ps, st = LuxCore.setup(MersenneTwister(42), layer)
q0, reconstruct = pack(default_vectorizer(), ps)
loss(q) = begin
    trajectory, _ = layer(nothing, reconstruct(q), st)
    last(trajectory.u)[1]^2 + sum(abs2, shooting_constraints(trajectory))
end
gradient = ForwardDiff.gradient(loss, q0)
@assert isapprox(only(gradient), 4exp(-4); rtol = 1e-7) # hide
gradient
```

Fixed initial states are promoted internally when tunable values introduce AD
scalar types. See the [optimization guide](optimization_contracts.md) for packing
and reconstruction in optimizer callbacks.

### ReverseDiff

The maintained serial configuration uses a fully specialized ODE function and
pass-through sensitivities. For an in-place right-hand side `rhs!`:

```julia
using SciMLBase, SciMLSensitivity, ReverseDiff

rhs!(dx, x, p, t) = (dx[1] = p[1] * x[1]; nothing)
f = ODEFunction{true, SciMLBase.FullSpecialize}(rhs!)
problem = ODEProblem(f, [1.0], (0.0, 2.0), [-1.0];
    sensealg = SciMLSensitivity.SensitivityADPassThrough(),
    abstol = 1e-10, reltol = 1e-10)
```

Construct the shooting layer from that problem. Pass-through lets the outer
backend differentiate the numerical solver directly and retains the solution
interface needed for segment handoff and trajectory assembly. This differs from
using a SciML adjoint algorithm and can have different memory and runtime costs.
The maintained tests use uncompiled ReverseDiff tapes.

!!! warning "The default ReverseDiff sensitivity route is a different configuration"
    The array-returning solve adapter does not retain the full solution interface
    required by the shooting layer. Use the tested pass-through configuration.
    Extracting a terminal matrix column alone does not recover actual saved times,
    solver metadata, or full trajectory support. The
    [archive](automatic_differentiation_history.md#ReverseDiff) records the failure
    and the relevant upstream adapter version.

### Zygote

The maintained tests use the default SciMLSensitivity path. Trajectory gathering
has a dedicated pullback; segment solve buffers use the Zygote extension when it
is loaded. These rules support Corleone's own assembly, not arbitrary mutation in
user callbacks.

Use allocating objective and constraint callbacks when relying on mutation-free
reverse AD. An in-place constraint callback requires support for its particular
mutation pattern; see [optimization contracts](optimization_contracts.md).

For threaded layers, the configured ChainRules map rule records block solves
serially and accumulates contributions from shared inputs. This establishes
correct gradients in the maintained tests, not parallel reverse execution. See
[parallel shooting](parallel_shooting.md) for execution and cache ownership.

## Fixed grids, parameters, and state

Numerical values in `ps` and fixed values in Lux state remain part of the
computation when supplied as differentiated inputs. “Fixed” means excluded from
the layer's trainable parameter tree; it does not detach a value from AD.

Grids, activity masks, sample selection, and symbolic bookkeeping are discrete
metadata. Tests establish derivatives on a fixed schedule. Changing dimensions,
types, or traced control flow can require new AD preparation; changing schedules
or expression layouts also requires re-preparing the optimization model.

A trajectory shares numerical data with its layer inputs and segment solutions.
Do not mutate those data between a forward call and its pullback. Parameter
queries and saved states have different ownership implications; see
[trajectories](@ref trajectory_guide).

For distributed ForwardDiff, load the same named objective or callable type and
model functions on every worker. Anonymous objective closures can change type
through serialization and produce ForwardDiff tag-ordering errors. Distributed
reverse AD is not established by those forward-mode checks.

## Validation sources

See `test/README.md` for maintained test groups and the
[validation archive](automatic_differentiation_history.md#Reproducing-checks-and-measurements)
for reproduction commands and dated measurements. Benchmark results do not extend
the support matrix above.
