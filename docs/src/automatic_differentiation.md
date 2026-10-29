# Automatic differentiation

AD compatibility depends on both the outer backend and the sensitivity strategy
used by the SciML solve. A passing forward solve does not establish that a
particular gradient configuration works.

## Maintained coverage

The `Differentiation` test group compares full-solver gradients with an analytic
reference at two inputs. Its losses include saved states and continuity residuals.
Inputs include shared parameters, free global initial values, interior shooting
values, and numerical values held in Lux state. The `Distributed` group separately
compares two-worker ForwardDiff gradients with serial results and finite differences.

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

```julia
using ComponentArrays, ForwardDiff

q0, reconstruct = pack(default_vectorizer(), ps)
loss(q) = begin
    trajectory, _ = layer(nothing, reconstruct(q), st)
    last(trajectory.u)[1]^2 + sum(abs2, shooting_constraints(trajectory))
end
gradient = ForwardDiff.gradient(loss, q0)
```

This fragment assumes an initialized `layer`, `ps`, and `st`; the
[optimization guide](optimization_contracts.md) explains the packing interface.
Fixed initial states are promoted internally when later tunable values introduce
AD scalar types. Caller-side initial-state promotion is not required by the
maintained tests.

### ReverseDiff

The maintained serial configuration uses a fully specialized ODE function and
pass-through sensitivities. For an in-place right-hand side `rhs!`:

```julia
using SciMLBase, SciMLSensitivity, ReverseDiff

f = ODEFunction{true, SciMLBase.FullSpecialize}(rhs!)
problem = ODEProblem(f, u0, tspan, p;
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

## Reproducing checks and measurements

From the repository root:

```sh
JULIA_NUM_THREADS=2 GROUP=Core julia --project=. -e 'using Pkg; Pkg.test()'
JULIA_NUM_THREADS=2 GROUP=Differentiation julia --project=. -e 'using Pkg; Pkg.test()'
GROUP=Distributed julia --project=. -e 'using Pkg; Pkg.test()'
```

`test/README.md` describes each group's environment and scope. The Core group
also checks trajectory construction, parameter reconstruction, resets, and
constraint derivatives. Run checks after changing these paths; old benchmark
results do not replace regression coverage.

`benchmark/shooting_ad.jl` measures forward loss calls and prepared gradients in
isolated backend processes. Its arguments include diagnostic configurations that
are expected to fail. See `benchmark/README.md` for dependencies, tolerances,
finite-difference checks, and the distinction between first-use and warmed costs.
The [validation archive](automatic_differentiation_history.md) preserves dated
measurements and numerical investigations; it is not a live performance ranking.
