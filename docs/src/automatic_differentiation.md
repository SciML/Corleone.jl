# Automatic differentiation backends

This page records the configurations tested with `ShootingLayer`, their known
limitations, and the design decisions behind differentiable trajectory assembly.
Compatibility depends on both the outer differentiation backend and the
`sensealg` used by the underlying SciML problem.

## Validation snapshot

The following results were obtained on **22 October 2026** with the full layer
returning `(trajectory, state)`, including `trajectory.residuals`. They describe
the tested configurations, not unconditional backend support.

| Backend | Control gradients | Trainable shooting-state gradients | Tested configuration | Maximum absolute gradient discrepancy |
|---|---|---|---|---:|
| ForwardDiff | Pass | Pass | Default ODE specialization and sensitivity settings | 1.25 × 10⁻⁷ |
| ReverseDiff | Pass | Pass | `FullSpecialize` + `SensitivityADPassThrough()`; initial-state promotion for the control-only case | 1.25 × 10⁻⁷ |
| Zygote | Pass | Pass | Default ODE specialization and sensitivity settings | 1.25 × 10⁻⁷ |
| Mooncake | Not tested | Pass | `FullSpecialize` + pass-through; segment times computed outside differentiation | 1.25 × 10⁻⁷ |
| Enzyme, forward mode | Pass | Not tested | `FullSpecialize` + pass-through; segment times computed outside differentiation; runtime activity and `Const` loss annotation | 6.00 × 10⁻¹⁰ |
| FiniteDiff | Pass | Not tested | Central differences | 1.60 × 10⁻⁹ |

FiniteDiff is numerical differentiation, rather than automatic differentiation.
"Not tested" does not indicate a failure. Compiled ReverseDiff tapes, Enzyme
reverse mode, Mooncake forward mode, Tracker, and other configurations are not
established by this snapshot.

All nine full-solver checks passed. They used an eight-interval Lotka ODE with
`Tsit5()` over `[0, 12]`, with `abstol = reltol = 1e-11`. The shooting-state cases
also varied physical parameters and initial/interior shooting states. The loss
combined the final third state with `0.3 * sum(r -> sum(abs2, r), trajectory.residuals)`.

Gradients were compared with central differences at two distinct inputs, using
steps `1e-5` and `5e-6` to check reference stability. The second input reused the
DifferentiationInterface preparation and exercised `gradient!`. Inputs were
checked for unintended mutation. These were correctness checks, not gradient
runtime benchmarks.

The snapshot predates the subsequent change to extract terminal states from
tracked matrices. After that change, the Core suite still passed all **183
assertions**, but the default ReverseDiff sensitivity path still failed on
`sol.alg`. Rerun the backend matrix after further changes; the historical passing
matrix does not replace regression checks on a new implementation.

The per-variable `InitialConditions` implementation has focused regression tests
in `test/core/initial_conditions_ad.jl` for ForwardDiff, ReverseDiff, and Zygote.
These check stored values, pass-through state values, residuals, and numeric values
held in Lux state. A full-solver ForwardDiff regression covers the first lifted
value occurring after the simulation starts: pass-through must retain the numeric
type required by later lifted values so the preallocated buffers remain compatible.
The other full-solver configurations in the historical matrix have not been rerun
for this representation.

## Configuring the solver and backend

### Full specialization and pass-through

The ReverseDiff, Mooncake, and Enzyme configurations above used the following
problem construction for an in-place right-hand side `rhs!`:

```julia
using SciMLBase, SciMLSensitivity

f = ODEFunction{true, SciMLBase.FullSpecialize}(rhs!)
prob = ODEProblem(f, u0, tspan, p;
    abstol = 1e-11, reltol = 1e-11,
    sensealg = SciMLSensitivity.SensitivityADPassThrough())
```

Pass-through bypasses SciML's sensitivity overrides and lets the outer backend
differentiate the numerical solver directly. It retains the solution object and
its saved metadata. It is a different differentiation strategy from using a
SciML adjoint algorithm, and the two strategies need not have the same memory or
runtime behavior. See the [SciML sensitivity interface](https://docs.sciml.ai/SciMLBase/v2.8/interfaces/Differentiation/).

### ForwardDiff and FiniteDiff

The tested DifferentiationInterface backends were `DI.AutoForwardDiff()` and
`DI.AutoFiniteDiff(; fdtype = Val(:central))`, where
`import DifferentiationInterface as DI`.

ForwardDiff passed both control-only and shooting-state cases with the default
problem configuration. The FiniteDiff backend check covered controls; central
differences were also used as the independent reference for the other cases.

### ReverseDiff

The passing configuration used `DI.AutoReverseDiff()` without a compiled tape,
together with full specialization and pass-through sensitivity.

For the control-only case, fixed initial states were promoted inside the
differentiated loss using the value-preserving expression
`[u + 0 * z[1] for u in prob.u0]`, where `z` is the active optimization vector.
The problem was remade with those initial states before applying the layer.
This makes the initial state compatible with the tracked arithmetic introduced
by the controls. The shooting-state case did not need this separate promotion.

!!! note "Default ReverseDiff sensitivity path is currently unsupported"
    DiffEqBase's default ReverseDiff solve rule returns a tracked array of state
    values instead of the full solution object. Its implementation converts the
    primal solution with `Array(out[1])`. Consequently, `.u`, `.t`, `.alg`, and
    other solution fields are unavailable on that return value.

    Extracting the terminal matrix column and passing its boundary time to the
    initial-condition layer fixes the first handoff problem. The subsequent
    solve must also obtain its algorithm from the layer, and trajectory assembly
    still needs the actual saved times and state samples. Updating terminal-state
    extraction alone does not establish default-path support.

The tested pass-through configuration remains the supported ReverseDiff route.
For the relevant upstream conversion, see the
[DiffEqBase 7.21.4 ReverseDiff extension](https://github.com/SciML/DiffEqBase.jl/blob/v7.21.4/ext/DiffEqBaseReverseDiffExt.jl).

### Zygote

`DI.AutoZygote()` passed both control-only and shooting-state cases with the
default problem configuration.

The original trajectory constructor collided with a LuxCore constructor rule
that dispatched broadly on `UnionAll`. The layer now calls a named builder, and
the public constructor has a constructor-specific configured ChainRules rule.
Rules for parametric constructors should dispatch on an appropriate `Type{...}`
signature, rather than `typeof(Constructor)`. See the
[ChainRules constructor guidance](https://juliadiff.org/ChainRulesCore.jl/stable/rule_author/writing_good_rules.html#Constructors).

Trajectory gathering uses a dedicated pullback instead of relying on
`Zygote.Buffer`. The existing segment-solver buffers still use the Zygote
extension when it is loaded. Zygote's
[limitations documentation](https://fluxml.ai/Zygote.jl/stable/limitations/)
describes mutation restrictions and cautions about `Buffer`.

### Mooncake and Enzyme

These cases computed the segment times before differentiation and reused them
while changing numerical parameters on the same grid. The benchmark used these
internal entry points:

```julia
segment_times = Corleone.get_timestops(layer, ps, st)
trajectory, next_st = layer((layer.problem, segment_times), ps, st)
```

This is distinct from preparing a gradient: all backend checks used
`DI.prepare_gradient`, while these two also prepared the segment schedule.
The current layer also caches segment times in `st.timestops` during setup;
ordinary forward calls reuse that cache. After changing schedules, refresh it
with `st = merge(st, (; timestops = Corleone.get_timestops(layer, nothing, st)))`.
Gradient preparations may also need rebuilding when dimensions, types, or traced
control flow change.

Mooncake used `DI.AutoMooncake()` and passed the shooting-state case. An initial
180-second worker limit was too short and was inconclusive. The full-trajectory
follow-up passed in approximately 290 seconds with a 900-second limit, including
package loading, compilation, and validation. These worker times are not warmed
gradient timings. In a separate run without trajectory construction, preparation
alone took approximately 158 seconds.

Enzyme used the following backend and passed the control-only case:

```julia
using Enzyme
import DifferentiationInterface as DI

backend = DI.AutoEnzyme(;
    mode = Enzyme.set_runtime_activity(Enzyme.Forward),
    function_annotation = Enzyme.Const)
```

The original trajectory construction raised an `IllegalTypeAnalysisException`.
Resolving the symbolic container once at layer construction and retaining its
concrete type removed that failure in the tested configuration. Runtime activity
was already enabled; enabling it alone was not the fix. See the
[Enzyme FAQ](https://enzymead.github.io/Enzyme.jl/dev/faq/) for activity and memory-layout requirements.

## Trajectory differentiation contract

The trajectory retains `layer`, `ps`, and `st` by reference. Saved state arrays
are also shared with the segment solutions. Parameter histories are evaluated
through `layer.parameter` and the retained `ps`/`st`; no separate parameter-history
struct or detached copy is stored. Mutating these objects changes later parameter
queries, so they must not be mutated between a forward evaluation and its pullback.

Assembly separates fixed-grid layout metadata from numerical state gathering.
The gathering pullback accumulates contributions into source samples, including
when an index occurs more than once. State values, residuals, and parameter values
remain differentiable. Symbolic metadata and the discrete layout are excluded
from the ChainRules differentiation path.

Validation covers numerical gradients on **fixed time grids**. It does not
establish gradients with respect to event times, control-knot positions, or
changes in grid topology. The full-solver matrix covers ODEs; it does not establish
DAE, SDE, GPU, or higher-order derivative support.

### Preserving metadata for array-returning solve rules

Actual saved times cannot generally be reconstructed from the outer control grid
or segment boundaries. `saveat`, callbacks, and other saving settings can affect
the samples saved by the solver. Internal `tstops` affect integration and need
not each appear as a saved sample. States and timestamps must correspond to the
same actual solve. DAE derivative values may need preserving as well.

If support for the default ReverseDiff sensitivity path becomes necessary, the
adapter belongs at the segment-solve boundary. It should perform one primal
solve, preserve that solve's actual metadata before array conversion, expose
differentiable numerical values, and connect those values to the solve pullback.
Metadata must remain consistent on subsequent evaluations and tape replay.
An extra solve solely to recover timestamps is not the intended design.

This adapter is deferred until a workload requires the array-returning adjoint
path. Prefer documenting the working configuration over maintaining a custom
solver differentiation rule without that requirement.

## Forward-pass measurements

The AD refactor was also compared with the preceding trajectory assembly on a
120-interval Lotka problem. Five rounds of 5,000 samples used one Julia thread;
these are warmed forward timings with **no gradient evaluation**.

| Mode | Trajectory assembly | Forward time | Allocated bytes/call | Allocations/call |
|---|---|---:|---:|---:|
| No AD backend loaded | Previous | 269.1 µs | 865,368 | 17,381 |
| No AD backend loaded | Current | 256.5 µs | 861,560 | 17,292 |
| Zygote loaded | Previous | 255.8 µs | 884,704 | 17,385 |
| Zygote loaded | Current | 252.8 µs | 879,888 | 17,295 |

The previous assembly was reconstructed on the same current solver. In a separate
assembly-only comparison with symbolic metadata cached in both versions, the old
loop took 1.72 µs and the new layout/gather took 1.68 µs. The new version allocated
6,768 bytes in 21 allocations, versus 6,048 bytes in 25 allocations for the old
loop. Thus the extra layout storage cost 720 bytes, with essentially equal
runtime. Allocated bytes are not peak or retained memory.

These measurements showed no full-forward regression; they do not rank backend
gradient performance. Loading Zygote changes the existing solver-buffer path, so
comparisons should record which optional backends are loaded.

## Regression checks and future validation

Run the Core test group from the repository root:

```sh
GROUP=Core julia --project -e 'using Pkg; Pkg.test()'
```

`test/core/trajectory_ad.jl` checks analytical gradients through the public
constructor and named builder with ReverseDiff and Zygote. It covers state,
residual, parameter-history, and combined losses at two inputs, continuous/reset
boundaries, and repeated-source-index accumulation.

The full backend matrix above was exploratory and is not part of the committed
Core suite. To repeat or extend it, use isolated processes per backend, reconstruct
the ODE/loss described above, and record the exact solver and backend configuration.
Compare central differences at multiple inputs, verify reference stability, reuse
gradient preparations at a second input, and check that inputs are unchanged.
For new solve-boundary adapters, include interior `saveat` samples and callbacks
and verify state/time correspondence. Record failures separately from timeouts
and untested combinations.

### Versions used in the validation snapshot

| Component | Version |
|---|---|
| Julia | 1.12.7 |
| SciMLBase | 3.58.0 |
| DiffEqBase | 7.21.4 |
| SciMLSensitivity | 7.119.12 |
| OrdinaryDiffEqTsit5 | 2.1.5 |
| LuxCore | 1.5.3 |
| DifferentiationInterface | 0.7.21 |
| ForwardDiff | 1.4.6 |
| ReverseDiff | 1.18.4 |
| Zygote | 0.7.13 |
| Mooncake | 0.5.63 |
| Enzyme | 0.13.216 |
| FiniteDiff | 2.33.0 |
