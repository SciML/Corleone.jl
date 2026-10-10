# AD compatibility

Run the registered backends with
`CORLEONE_TEST_GROUP=AD julia --project=lib/CorleoneBase -e 'using Pkg; Pkg.test()'`
from the repository root. The AD group resolves its isolated `test/ad` environment
against the checked-out package. Default `Pkg.test()` runs only lightweight Core;
`CORLEONE_TEST_GROUP=QA` selects the isolated QA environment.
The registered backends are ForwardDiff, uncompiled ReverseDiff,
Zygote, Mooncake reverse mode, Mooncake forward mode, and a finite-difference
baseline. SciMLSensitivity is loaded, and
ReverseDiff uses
`SciMLSensitivity.SensitivityADPassThrough()` to trace the inner solves while
retaining full solution metadata for transitions and return-code checks. The
default ReverseDiff adjoint rule returns only a tracked state array, not a full
solution. Compiled ReverseDiff tapes are outside this suite's scope.

For out-of-place ODE stages, the driver normalizes state and parameter arrays
with `ArrayInterface.aos_to_soa` before solving. This keeps tracked states and
broadcast derivatives in the same array representation without changing the
model's in-place/out-of-place form.

Zygote uses the `CorleoneBaseZygoteExtension` pullback for `solve!`, with
SciMLSensitivity's default inner-solve rules. The pullback differentiates
stage accumulation through an internal `Zygote.Buffer`, including transitions
and saved endpoints, while preserving the primal iterator and buffer identity.
Discrete stage indices and return-code decisions carry no derivative. The internal buffer is initialized
from the existing solutions and frozen with `copy` only after solving finishes.
Extension-local identity pullbacks normalize SciML state cotangents to
structural solution cotangents with the full solution field set, so Buffer can
accumulate saved-endpoint and transition-metadata gradients. Supplemental
rule tests cover early termination, failed stages, preallocated buffers, zero
cotangents, and direct
iterator metadata gradients. Structural field-access and iterator-constructor
pullbacks avoid double accumulation through Zygote's mutable-object gradient
cache. The initialization pullback filters keyword names without array mutation;
keyword values remain differentiable.

The reusable suite accepts any DifferentiationInterface backend:

Each backend runs all four combinations: linear and nonlinear models, each
with out-of-place `f(u, p, t)` and in-place `f!(du, u, p, t)` dynamics. They share
the same loss, analytic references, and primal/prepared/unprepared checks. The
primal checks also verify the problem's in-place flag after every transition.

```julia
using DifferentiationInterface, ForwardDiff, Zygote, SciMLSensitivity
include("suite.jl") # use this file's directory, or an absolute path
using .SequentialADTests

test_sequential_ad(AutoZygote(); solve_kwargs = (; sensealg = ForwardDiffSensitivity()))
```

Load/install the chosen backend and sensitivity packages in the calling test
environment. Available backends with incompatible sequential solves produce test
errors; they are not silently skipped. Supply backend-specific tolerances only
when justified by its numerical differentiation method.

Both cases differentiate `x = [u₁, p₁, p₂, u₃]`. Stage one applies `u₁` and `p₁`
via solve-time remake keywords. Stage two carries the previous endpoint and
changes to `p₂`. Stage three resets to `u₃` and retains `p₂`. Time spans are
`(0, 1)`, `(1, 2)`, and `(2, 3)`, with a fixed three-stage terminal.

The weighted squared-error loss uses every stage endpoint. Closed-form endpoints
and hand-derived Jacobians provide independent gradient references for both
`du/dt = -p*u` and `du/dt = -p*u²`. Prepared evaluations alternate inputs and
repeat the first input to expose stale tapes or leaked mutable state.

## Parallel ForwardDiff and ReverseDiff contracts

The AD group also registers `parallel.jl`, backed by the reusable
`ParallelADTests.test_parallel_ad(backend, modes; solve_kwargs, gradient_atol,
gradient_rtol)` in `parallel_suite.jl`. ForwardDiff (all three ensemble modes)
and uncompiled ReverseDiff (serial and threaded modes, with
`SciMLSensitivity.SensitivityADPassThrough()` and `safetycopy = false` supplied
via `solve_kwargs`) are registered for parallel AD; the other existing backends
remain sequential-only. The pass-through matches the sequential suite, and
skipping SciMLBase's default per-simulation template deepcopy keeps ReverseDiff's
tape-bound template from overflowing the stack inside `deepcopy` on Julia 1.10
and 1.11 (the harness does not need the copy: `prob_func` builds fresh problems
and nothing mutates the template). ReverseDiff is not registered for
`EnsembleDistributed`: a `ReverseDiff.GradientTape` cannot survive Distributed's
pmap serialization (gradients come back zero and the orphaned tape structures
segfault the workers); see `PARALLEL_RESULTS.md` for the reproducer and unblock
path. Additional parallel backends can use the same harness after loading their
packages on the driver and workers. Availability and differentiation failures
are assertions or errors, never skips.

The harness crosses direct `ODEProblem` and wrapped three-stage
`SequentialProblem` templates with linear/nonlinear and in-place/out-of-place
dynamics under `EnsembleSerial`, `EnsembleThreads`, and (for ForwardDiff)
`EnsembleDistributed`. Out-of-place direct-template problems are normalized
with `CorleoneBase.prepare_stage_problem` before the ensemble machinery solves
them, mirroring the sequential driver; it is a no-op for untraced inputs, so
ForwardDiff coverage is unaffected.
Three trajectories use `zᵢ = (1 + i/10)x + i/20`, giving distinct initial states
and parameters. Direct templates differentiate `[u₁, p₁]`; sequential templates
differentiate `[u₁, p₁, p₂, u₃]`. Every differentiated coordinate contributes to
the scalar, trajectory-weighted endpoint loss. References reuse the sequential
suite's closed-form endpoints and hand-derived Jacobians, applying the affine
trajectory chain rule analytically, without AD or numerical differentiation.

Primal checks cover every trajectory and stage, return codes, in-place flags,
times, initial states, parameters, endpoints, and aggregate loss. Unprepared
and prepared gradients cover two distinct inputs and a repeated first input,
shape, finiteness, analytic agreement, and unchanged inputs (including
preparation). Tolerances are unchanged from the sequential ForwardDiff suite:
ODE `abstol = reltol = 1e-11`, value/endpoint `atol = rtol = 1e-9`, and gradient
`atol = 1e-7`, `rtol = 1e-6`.

The driver starts two local workers with two threads each and cleans up only
those workers in `finally`. Execution-location probes assert that threaded
work uses multiple thread IDs and distributed work uses multiple non-driver
process IDs. A single-threaded AD entry point launches a four-threaded
subprocess instead of skipping threaded tests. The fixture module is loaded
under the same `Main.ParallelADTests` identity on driver and workers, including
when the registered group runs inside SafeTestsets.

For a fresh, recorded environment, run from the repository root:

```sh
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.10 --startup-file=no --threads=4 lib/CorleoneBase/test/ad/verify_parallel.jl
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.11.7~x64 --startup-file=no --threads=4 lib/CorleoneBase/test/ad/verify_parallel.jl
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.12 --startup-file=no --threads=4 lib/CorleoneBase/test/ad/verify_parallel.jl
```

The script copies the AD group's declarations into a persistent scratch project
under `/tmp/opencode`, develops the checked-out CorleoneBase, records the full
resolved manifest, Julia version, thread counts, worker IDs, and execution
locations, then runs the parallel contracts. It does not edit repository
metadata. See `PARALLEL_RESULTS.md` for observed results and regression commands.
