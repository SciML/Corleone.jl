# Parallel AD verification

Commands run from `/home/alcap/Projects/Corleone.jl.worktrees/CJM-CorleoneBase`
on Linux x86_64, against the checked-out `lib/CorleoneBase` (0.1.0).
Only test files and the `CorleoneBaseZygoteExtension` were changed; no `src`
or package-metadata change was needed. Pre-existing unrelated changes were left
untouched.

The statement above describes the earlier phases. The Mooncake phase initially
used an explicitly test-loaded `CorleoneBaseMooncakeExtension`; see the appended
Mooncake section. It is now registered in the package metadata (`Mooncake`
weakdep + extension entry + `Mooncake = "0.5"` compat), so the rules load
automatically whenever CorleoneBase and Mooncake are co-loaded.

This document grew by phase: first ForwardDiff, then uncompiled ReverseDiff, then
Zygote. Each phase's results stand on its own; counts below are per phase.
Zygote was added in the `Add parallel Zygote` commit, which is exactly the
three-file change (extension, parallel driver, README) described in
`README.md`.

## Environments and commands

Fresh environments were created using:

```sh
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.10 --startup-file=no --threads=4 lib/CorleoneBase/test/ad/verify_parallel.jl > /tmp/opencode/parallel-ad-110.log 2>&1
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.11.7~x64 --startup-file=no --threads=4 lib/CorleoneBase/test/ad/verify_parallel.jl > /tmp/opencode/parallel-ad-111.log 2>&1
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.12 --startup-file=no --threads=4 lib/CorleoneBase/test/ad/verify_parallel.jl > /tmp/opencode/parallel-ad-112-final.log 2>&1
```

Each log records the full resolved manifest. The final verification (with both
backends registered) reused the same script on each version, producing the
persistent environments `/tmp/opencode/parallel-ad-1.10.12-wK2fCv`,
`/tmp/opencode/parallel-ad-1.11.7-Dws0eL`, and
`/tmp/opencode/parallel-ad-1.12.7-DqGZaS`, respectively, with logs
`parallel-ad-rd-110-final.log`, `parallel-ad-rd-111-final.log`, and
`parallel-ad-rd-112-final.log`. All resolve DifferentiationInterface 0.7.21,
ForwardDiff 1.4.6, ReverseDiff 1.18.4, SciMLSensitivity 7.119.12, SciMLBase
3.57.0, CommonSolve 0.2.14, OrdinaryDiffEqTsit5 2.1.5, SafeTestsets 0.1.0, and
SciMLTesting 2.13.2. Their manifests record the matching Julia version.
CorleoneBase is path-developed against this checkout, not a registered release.

The first, ForwardDiff-only verification used the same commands (persistent
environments `parallel-ad-1.10.12-p4pOmu`, `parallel-ad-1.11.7-faUxsB`, and
`parallel-ad-1.12.7-V0E1Gv`, logs `parallel-ad-1{10,11}.log` and
`parallel-ad-112-final.log`); its ForwardDiff results are unchanged by the
ReverseDiff registration.

After stabilizing the harness, focused verification reused the resolved
environments:

```sh
julia +1.10 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.10.12-p4pOmu -e 'include("lib/CorleoneBase/test/ad/parallel.jl")' > /tmp/opencode/parallel-ad-rd-110-fix.log 2>&1
julia +1.11.7~x64 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.11.7-faUxsB -e 'include("lib/CorleoneBase/test/ad/parallel.jl")' > /tmp/opencode/parallel-ad-rd-111-fix.log 2>&1
julia +1.12 --startup-file=no --threads=4 --project=/tmp/opencode/cb-ad -e 'using SafeTestsets; @safetestset "Parallel AD registered isolation" begin; include("lib/CorleoneBase/test/ad/parallel.jl"); end' > /tmp/opencode/parallel-ad-isolated-rd-112.log 2>&1
```

`cb-ad` has the same relevant resolved versions as the fresh 1.12 environment.
Each backend's suite checks analytic/primal/AD contracts — ForwardDiff 3,168,
ReverseDiff 2,112, and Zygote 3,168 — plus nine driver checks for threads,
workers, availability, and actual execution locations.

The Zygote phase reused fresh path-developed environments
`/tmp/opencode/zyg-1.10.12-UKGBVq`, `/tmp/opencode/zyg-1.11.7-O8GZpL`, and
`/tmp/opencode/zyg-1.12.7-SVUk56` (same resolved versions as above, each
recording its matching Julia version), driven by the committed extension via

```sh
julia +1.10 --startup-file=no --threads=4 --project=/tmp/opencode/zyg-1.10.12-UKGBVq -e 'include("lib/CorleoneBase/test/ad/parallel.jl")' > /tmp/opencode/zyg_full_driver_110.log 2>&1
julia +1.11.7~x64 --startup-file=no --threads=4 --project=/tmp/opencode/zyg-1.11.7-O8GZpL -e 'include("lib/CorleoneBase/test/ad/parallel.jl")' > /tmp/opencode/zyg_full_driver_111.log 2>&1
julia +1.12 --startup-file=no --threads=4 --project=/tmp/opencode/zyg-1.12.7-SVUk56 -e 'include("lib/CorleoneBase/test/ad/parallel.jl")' > /tmp/opencode/zyg_full_driver_112.log 2>&1
```

The sequential AD suite was re-run unchanged on all three versions to confirm
the extension change did not disturb sequential Zygote (logs
`zyg_seqad_{110,111,112}.log`).

## Observed parallel results

| Julia | Analytic/primal/AD checks | Actual threaded IDs | Distributed OS process IDs |
| --- | --- | --- | --- |
| 1.10.12 | 8,448 passed: 3,168 ForwardDiff + 2,112 ReverseDiff + 3,168 Zygote | 1, 2, 3, 4 (driver PID 972170) | 972266, 972269 |
| 1.11.7 | 8,448 passed: 3,168 ForwardDiff + 2,112 ReverseDiff + 3,168 Zygote | 1, 2, 3, 4 (driver PID 974058) | 974183, 974187 |
| 1.12.7 | 8,448 passed: 3,168 ForwardDiff + 2,112 ReverseDiff + 3,168 Zygote | 2, 3, 4, 5 (driver PID 963566) | 963646, 963650 |

All runs use four default-pool driver threads and two newly started worker
processes, each with two default-pool threads. Julia 1.12 also has an
interactive thread, hence its default-pool IDs start at two. The execution
probes check observed locations, not merely configured resources. Serial
execution stays on the driver; distributed execution uses both remote processes.

The registered ForwardDiff coverage is 24 combinations: three ensemble modes ×
two template kinds × two models × two dynamics forms. ReverseDiff adds 16
combinations: the same crossings under `EnsembleSerial` and `EnsembleThreads`
with the pass-through sensealg. Zygote adds the full 24 combinations again,
under all three ensemble modes. Each combination has three distinct
trajectories, two inputs, unprepared gradients, and prepared evaluations of
both inputs then the first again. No parallel assertion is skipped or marked
broken. Numerical tolerances match the sequential ForwardDiff suite exactly;
see `README.md`.

## Investigations retained

The first 1.12 and 1.10 runs passed 3,288 checks. The direct-template input was
then tightened from four coordinates (two unused in a single-stage ODE) to
`[u₁, p₁]`, so every differentiated input contributes to the loss; this explains
the final 3,168 count, without dropping any model, trajectory, mode, or AD path.

The first 1.11 run passed 2,888 checks but errored in all eight distributed
gradient combinations (16 testset errors). Workers reported
`UndefVarError: #38#40 not defined in Main.ParallelADTests`: the source file had
changed while the driver and workers were loading it, leaving different closure
definitions. This failure is retained in `parallel-ad-111.log`, not skipped.
The stable-source rerun passed all 3,168 contracts on 1.11.7 (see
`parallel-ad-111-final.log`). Subsequent verification runs use the stable fixture
module loaded under the same
`Main.ParallelADTests` identity on every process. SafeTestsets isolation is also
explicitly verified; fixtures are not defined under a driver-only generated
testset module.

The first direct Core invocation in `cb-ad` failed because ArrayInterface was
not a direct dependency of that AD environment. It was rerun in the Core
environment, rather than changing package metadata. The first 1.10 Core run
then exposed a stale test expecting a pre-1.12 `MethodError`. Current source
already contains the compatible callback binder. The test now requires the
same existing successful parallel contracts on all versions, preserving all
success assertions and strengthening older-version coverage. No source change
or weaker numerical assertion was used.

## ReverseDiff sensitivity configuration and the distributed limitation

ReverseDiff registers with the same harness as ForwardDiff, using
`SciMLSensitivity.SensitivityADPassThrough()` — the only configuration that
preserves full solution metadata while ReverseDiff traces the inner solves.
The default adjoint paths return only a tracked state array and break the
suite's metadata and return-code contracts identically for every tested
sensealg (captured in `rd-sensealg-serial-rd.log`):

- direct out-of-place templates: `FieldError: type ReverseDiff.TrackedArray
  has no field 'u'` when the loss reads solution fields;
- wrapped sequential templates: `MethodError: Cannot convert an object of type
  ReverseDiff.TrackedArray{Float64, Float64, 2, Matrix{Float64},
  Matrix{Float64}} to an object of type SciMLBase.ReturnCode.T` at the retcode
  check.

This holds for the default sensealg, `ReverseDiffAdjoint()`, and
`InterpolatingAdjoint(autojacvec = ReverseDiffVJP())`; only pass-through keeps
the contracts.

ReverseDiff also requires `safetycopy = false` for the ensemble solves.
SciMLBase's ensemble `batch_func` deep-copies the template problem per
simulation when `EnsembleProblem.safetycopy` is true (the default; CorleoneBase
does not set it). With ReverseDiff-traced inputs that deepcopy recurses through
the tape graph reachable from the template; on Julia 1.10 and 1.11 it reaches
the stack limit (`StackOverflowError` inside `Base.deepcopy_internal`, ~18.6k
frames) after spending minutes per gradient evaluation, and the wrapped
in-place templates under `EnsembleSerial` fail both gradient testsets. The
first 1.10 and 1.11 ReverseDiff runs with the default safetycopy passed 2,032
checks and errored in four gradient testsets after 20m and 11m respectively
(retained in `parallel-ad-rd-110.log` and `parallel-ad-rd-111.log`); Julia 1.12
passes the same solves because its `deepcopy_internal` skips recursing into
isbits fields. The harness does not need the per-simulation copy: `prob_func`
builds fresh problems and nothing mutates the template, so the ReverseDiff
registration passes `safetycopy = false` (an ensemble keyword CorleoneBase's
`ParallelProblem` solve documents). ForwardDiff keeps the default. All three
Julia versions resolve identical package versions, so this is a Julia deepcopy
difference, not a package-version difference.

`EnsembleDistributed` is not registered for `AutoReverseDiff`. SciMLBase's
ensemble solve pmap-serializes each trajectory closure to the workers, and a
closure traced by ReverseDiff carries its `GradientTape`; Distributed's
serialization cannot keep the worker-side tape connected to the driver's
tracked input, so the tape's contribution is silently lost. The full
distributed ReverseDiff run shows zero gradients and a stack overflow that ends
in a segfault (exit 139). No test-local change can reconnect the tape across a
process boundary. A minimal reproducer, independent of CorleoneBase (saved at
`/tmp/opencode/tape_serialize_probe.jl`):

```julia
using Distributed
using ReverseDiff
addprocs(1; exeflags = `--startup-file=no`)
@everywhere using ReverseDiff
function loss(x)
    y = remotecall_fetch(workers()[1], x) do xw
        return sum(xw .^ 2)
    end
    return y + 1
end
ReverseDiff.gradient(loss, [1.0, 2.0])  # [0.0, 0.0], not [2.0, 4.0]
```

Unblocking distributed ReverseDiff would require an upstream change —
ensemble-level reverse-mode handling in SciMLBase or tape-serialization
support in ReverseDiff — or a CorleoneBase ReverseDiff extension with a
worker-side pullback; both are outside this test-only change.

## Zygote reverse-mode and the ensemble adjoint rule

Zygote registers with the same harness as ForwardDiff and ReverseDiff, across
all three ensemble modes and both template kinds, and needs no solver-time
`sensealg` and no `solve_kwargs`. Two native Zygote limitations make plain
instrumentation unusable for the parallel ensemble, so the
`CorleoneBaseZygoteExtension` adjoins the real ensemble solve directly.

- **Zygote reverse-mode is not thread-safe.** Under `EnsembleThreads` the
  wrapped `SequentialProblem` trajectories run the extension's `solve!` pullback
  (`rrule_via_ad` + `Zygote.Buffer`) concurrently; the recorder races and
  intermittently returns wrong gradients. In a 10-iteration probe across all
  four models and both inputs, 7 of 80 threaded wrapped evaluations were wrong
  (max error up to 2.4) with the native trace. Direct `ODEProblem` templates (via
  SciMLSensitivity's rule) were unaffected: 0 of 80 threaded evaluations failed.
- **Zygote's shadow/pullback structures cannot survive Distributed's pmap
  serialization.** Under `EnsembleDistributed`, SciMLBase's `batch_func`
  serializes a `Zygote.Pullback`-typed result back to the driver and the workers
  crash; this is the same process-boundary wall that blocks ReverseDiff's tape.

The extension fixes both with one rule for `CommonSolve.solve(::AbstractParallelProblem,
…)`: it runs the requested-mode ensemble solve as an **opaque forward** (outside
Zygote's recorder, so threaded/distributed execution is genuine and the
execution-location probes still observe it) and in the **pullback** re-differentiates
the actual trajectory problems serially via `rrule_via_ad` on a helper that
rebuilds the same `EnsembleProblem` and solves it with `EnsembleSerial`.
Ensemble solutions are mode-independent, so the adjoint reproduces the forward's
primal values exactly while computing every trajectory's real trajectory-specific
gradient (`prob_func`/closure captures included) without racing or serializing a
tape. The minimal probe that first demonstrated the threaded race
(`/tmp/opencode/zyg_race.jl`) re-passes 0/80 (and 0/120 in a second probe) once
the rule is loaded; the distributed reproducer (`/tmp/opencode/zyg_dist2.jl`)
passes all eight combos on real workers.

Zygote parallel verification: **3,168/3,168** analytic/primal/AD checks on each
of Julia 1.10.12, 1.11.7, and 1.12.7, plus nine driver checks. Sequential AD is
unchanged: **2,143** total per version (six backends at 344 each, 72 supplemental
Zygote rules, seven top-level assertions), confirming the extension change does
not disturb sequential Zygote (`solve!` still uses its pullback).

## Core regression verification

Core passed 330 sequential contracts, four parallel construction/interface
contracts, and 15 parallel workflow contracts on each of Julia 1.10.12, 1.11.7,
and 1.12.7 (349 assertions within the reported testsets, plus passing top-level
source/extension checks). The sequential file was unchanged. Core loads no Zygote,
so the extension stays unloaded there (asserted by the core source check).
Exact commands:

```sh
julia +1.10 --startup-file=no --threads=4 --project=/tmp/opencode/cb-core-110 -e 'include("lib/CorleoneBase/test/core/sequential.jl"); include("lib/CorleoneBase/test/core/parallel.jl")' > /tmp/opencode/parallel-core-110-final.log 2>&1
julia +1.11.7~x64 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.11.7-faUxsB -e 'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.add(["ArrayInterface", "OrdinaryDiffEqTsit5", "CommonSolve", "SciMLBase", "Test"]); include("lib/CorleoneBase/test/core/sequential.jl"); include("lib/CorleoneBase/test/core/parallel.jl")' > /tmp/opencode/parallel-core-111-final.log 2>&1
julia +1.12 --startup-file=no --threads=4 --project=/tmp/opencode/cb-core -e 'include("lib/CorleoneBase/test/core/sequential.jl"); include("lib/CorleoneBase/test/core/parallel.jl")' > /tmp/opencode/parallel-core-112-final.log 2>&1
```

The 1.11 command builds a fresh Core scratch environment, recorded in its log;
the other two use existing checked-source Core environments. All resolve
ArrayInterface 7.30.2, CommonSolve 0.2.14, SciMLBase 3.57.0, and
OrdinaryDiffEqTsit5 2.1.5. The 1.12 run preceded the older-version branch removal;
its 15 workflow assertions were identical.

## Single-thread entry point

```sh
julia +1.12 --startup-file=no --threads=1 --project=/tmp/opencode/cb-ad -e 'using SafeTestsets; @safetestset "Parallel AD single-thread entry" begin; include("lib/CorleoneBase/test/ad/parallel.jl"); end' > /tmp/opencode/parallel-ad-single-entry-rd-112.log 2>&1
```

Passed: the parent asserts the child process succeeds; the child passes all
3,168 ForwardDiff, 2,112 ReverseDiff, and 3,168 Zygote contracts with four driver threads. Its
probes observe thread IDs 2, 3, 4, 5 on driver PID 963566 and remote PIDs
for the distributed mode. Thus ordinary single-threaded test invocation still
exercises actual threaded and distributed execution rather than substituting
serial coverage.

## Registered AD group regression

```sh
JULIA_PKG_PRECOMPILE_AUTO=0 CORLEONE_TEST_GROUP=AD julia +1.12 --startup-file=no --threads=4 --project=/tmp/opencode/cb-ad -e 'include("lib/CorleoneBase/test/runtests.jl")' > /tmp/opencode/parallel-ad-group-rd-112.log 2>&1
```

Passed on Julia 1.12.7: unchanged Sequential AD **2,143/2,143**, Parallel AD
**8,458/8,458** (3,168 ForwardDiff + 2,112 ReverseDiff + 3,168 Zygote contracts
plus the driver's worker/thread/availability/execution-location checks); total
**10,601** with no skips, broken assertions, or errors. This uses SciMLTesting's
isolated AD environment `/tmp/jl_sciml_group_env_LyhWrs`, resolved against the
checked-out package with the same relevant versions above, and exercises the
SafeTestsets registrations. The unchanged sequential driver covers ForwardDiff,
ReverseDiff, Zygote, Mooncake reverse/forward, FiniteDiff, and supplemental
Zygote rules. Parallel ForwardDiff, ReverseDiff, and Zygote therefore also pass
with SciMLSensitivity and the other existing AD packages loaded. Parallel AD
probes observe default-pool thread IDs (driver process) and the two remote
worker OS PIDs for distributed mode inside the SafeTestsets-isolated run.

The group log retains Julia's warnings about precompiled dependencies differing
from already loaded modules during environment activation. They are not test
failures; the separately started fresh-environment parallel runs independently
verify the recorded resolves.

Supplemental unchanged sequential AD verification also passes on Julia 1.10.12
and 1.11.7:
six backend suites at 344 assertions each, 72 supplemental Zygote rule checks,
and seven top-level source/availability assertions (2,143 total per version).
Commands:

```sh
julia +1.10 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.10.12-p4pOmu -e 'include("lib/CorleoneBase/test/ad/runtests.jl")' > /tmp/opencode/parallel-sequential-ad-rd-110.log 2>&1
julia +1.11.7~x64 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.11.7-faUxsB -e 'include("lib/CorleoneBase/test/ad/runtests.jl")' > /tmp/opencode/parallel-sequential-ad-rd-111.log 2>&1
```

## Scope of earlier phases

ForwardDiff parallel support (all three ensemble modes), uncompiled ReverseDiff
parallel support (serial and threaded modes), and Zygote parallel support (all
three ensemble modes, via the extension's ensemble adjoint rule) are claimed.
The backend-neutral harness is ready for later explicitly selected DI backends,
but those backends' parallel support is not required or inferred here. These
checks do not certify GPUs, external cluster workers, stochastic
differentiation, compiled ReverseDiff tapes, distributed ReverseDiff evaluation
(see the limitation above), or dependency versions other than the recorded
resolves. Zygote parallel registration specifically requires the committed
extension rule; without it (native trace) threaded wrapped results race and
distributed workers crash, as documented above.

## Mooncake reverse-mode phase

Initial native-trace commands (no Mooncake solve-boundary rule):

```sh
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.10 --startup-file=no --threads=4 lib/CorleoneBase/test/ad/verify_parallel.jl > /tmp/opencode/mooncake-1.10.log 2>&1
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.11.7~x64 --startup-file=no --threads=4 lib/CorleoneBase/test/ad/verify_parallel.jl > /tmp/opencode/mooncake-1.11.log 2>&1
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.12 --startup-file=no --threads=4 lib/CorleoneBase/test/ad/verify_parallel.jl > /tmp/opencode/mooncake-1.12.log 2>&1
```

Resolved environments are `/tmp/opencode/parallel-ad-1.10.12-Vg4zko`,
`/tmp/opencode/parallel-ad-1.11.7-gO7XfS`, and
`/tmp/opencode/parallel-ad-1.12.7-c6AdEX`. Each has a persistent Project and
Manifest and develops CorleoneBase 0.1.0 from this checkout. All three resolve
Mooncake 0.5.63, DifferentiationInterface 0.7.21, SciMLBase 3.57.0,
SciMLSensitivity 7.119.12, DiffEqBase 7.21.3, OrdinaryDiffEqTsit5 2.1.5,
ForwardDiff 1.4.6, ReverseDiff 1.18.4, and Zygote 0.7.13.

On 1.10/1.11, ForwardDiff passed 3,168 checks before Mooncake ran. Native
Mooncake passed 2,304 primal checks but errored in all 48 gradient testsets
(unprepared and prepared across 24 combinations), including serial execution.
The first error is `Mooncake.UnhandledLanguageFeatureException` in
`Base.CoreLogging.with_logstate`: reverse-mode differentiation does not support
`try/catch/finally`. No failing case was skipped. A focused reproducer, from the
repository root in either environment, is:

```julia
using Mooncake, SciMLSensitivity, DifferentiationInterface, SciMLBase
include("lib/CorleoneBase/test/ad/parallel_suite.jl")
P = ParallelADTests
f = x -> P.ensemble_loss(P.LinearCase{false}(), x, false, EnsembleSerial();
    solve_kwargs = (; safetycopy = false))
value_and_gradient(f, AutoMooncake(), [1.2, 0.4])
```

The first 1.12 invocation warned of a stack overflow and segfaulted (exit 139).
`coredumpctl info 13788` confirms SIGSEGV in the Julia driver; its logged stack is
unsymbolized. The log contains environment-activation warnings about already
loaded dependency versions differing from the newly resolved manifest; it does
not establish the precise cause of the stack overflow. The kernel journal around
the crash has no OOM entries. Verification now launches a fresh child after
resolution to eliminate that environment ambiguity; the native backend failure
on the other versions independently establishes the need for an ensemble rule.

The attempted solve-boundary approach uses the real requested-mode forward
ensemble, then directly maps the real trajectory hook and solves in serial,
avoiding scheduler/logging instrumentation. It differentiates those inner solves
with Mooncake, including input-dependent problem construction and transitions;
the rule never calls analytic references. Only deterministic, mode-independent,
default-output/default-reduction ensembles are covered. `safetycopy = false` is
required; custom output/reduction, seed/RNG, and batching keywords are explicitly
rejected. Loading is explicit on driver and workers because automatic package
extension registration would require forbidden metadata changes.

The focused rule-loaded probe on Julia 1.10 returned value
`1.7552838932325958` and gradient `[3.0691566576017815, -4.9047424269932]`.
This single probe alone is not evidence of full parallel support; the full
matrix is verified below.

Rule-loaded full-driver results (all three commands exited successfully):

| Julia | Mooncake | Preserved backends | Thread IDs / distributed OS PIDs |
| --- | --- | --- | --- |
| 1.10.12 | 3,168/3,168 | ForwardDiff 3,168; ReverseDiff 2,112; Zygote 3,168 | 1–4 / 43267, 43270 |
| 1.11.7 | 3,168/3,168 | ForwardDiff 3,168; ReverseDiff 2,112; Zygote 3,168 | 1–4 / 43295, 43299 |
| 1.12.7 | 3,168/3,168 | ForwardDiff 3,168; ReverseDiff 2,112; Zygote 3,168 | 2–5 / 43305, 43309 |

Each Mooncake run covers all 24 combinations, including every analytic primal,
prepared/unprepared gradient, changed-input, repeat, and input-preservation
assertion. No tolerances or existing assertions were changed. The driver uses
four default-pool threads and two workers with two threads each.

A non-fixture default-hook probe also passed on Julia 1.10
(`/tmp/opencode/mooncake-template-dependencies.log`), confirming dependence on
the template's state and parameters rather than only the fixture's captured
trajectory hook. Reproduce in the 1.10 scratch environment:

```julia
using Mooncake, ForwardDiff, SciMLSensitivity, DifferentiationInterface,
    SciMLBase, CorleoneBase, CommonSolve, OrdinaryDiffEqTsit5, Test
include("lib/CorleoneBase/ext/CorleoneBaseMooncakeExtension.jl")
function loss(x)
    ode = ODEProblem((u, p, t) -> p[1] .* u .+ p[2],
        [x[1]], (0.0, 0.8), [x[2], x[3]])
    ensemble = solve(ParallelProblem(ode), Tsit5(), EnsembleSerial();
        trajectories = 2, safetycopy = false, abstol = 1e-11, reltol = 1e-11,
        save_everystep = false)
    return sum(sol.u[end][1]^2 for sol in ensemble.u)
end
x = [0.7, 0.3, 0.2]
original = copy(x)
a = value_and_gradient(loss, AutoMooncake(), x)
b = value_and_gradient(loss, AutoForwardDiff(), x)
@test a[1] ≈ b[1]
@test isapprox(a[2], b[2]; atol = 1e-7, rtol = 1e-6)
@test x == original
```

Mooncake returned `2.2928276967982333` and gradient
`[5.444542331131, 3.3711137426877853, 3.8723788090236897]`; ForwardDiff agreed
to approximately 2e-13. This supplements, not replaces, the analytic matrix.

Rule-loaded full-driver commands reuse the resolved manifests:

```sh
julia +1.10 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.10.12-Vg4zko lib/CorleoneBase/test/ad/parallel.jl > /tmp/opencode/mooncake-full-1.10.log 2>&1
julia +1.11.7~x64 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.11.7-gO7XfS lib/CorleoneBase/test/ad/parallel.jl > /tmp/opencode/mooncake-full-1.11.log 2>&1
julia +1.12 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.12.7-c6AdEX lib/CorleoneBase/test/ad/parallel.jl > /tmp/opencode/mooncake-full-1.12.log 2>&1
```

Fresh regressions (Core before loading AD packages; sequential AD with the new
Mooncake rule loaded) use:

```sh
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.10 --startup-file=no --threads=4 lib/CorleoneBase/test/ad/verify_regressions.jl > /tmp/opencode/mooncake-regressions-1.10.log 2>&1
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.11.7~x64 --startup-file=no --threads=4 lib/CorleoneBase/test/ad/verify_regressions.jl > /tmp/opencode/mooncake-regressions-1.11.log 2>&1
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.12 --startup-file=no --threads=4 lib/CorleoneBase/test/ad/verify_regressions.jl > /tmp/opencode/mooncake-regressions-1.12.log 2>&1
```

Isolation and entry-point verification uses the final driver, including explicit
upstream extension-loading checks on driver/workers and a post-`rmprocs` cleanup
assertion:

```sh
julia +1.10 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.10.12-Vg4zko -e 'using SafeTestsets; @safetestset "Parallel AD isolation" begin; include("lib/CorleoneBase/test/ad/parallel.jl"); end' > /tmp/opencode/mooncake-safe-1.10.log 2>&1
julia +1.11.7~x64 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.11.7-gO7XfS -e 'using SafeTestsets; @safetestset "Parallel AD isolation" begin; include("lib/CorleoneBase/test/ad/parallel.jl"); end' > /tmp/opencode/mooncake-safe-1.11.log 2>&1
julia +1.12 --startup-file=no --threads=1 --project=/tmp/opencode/parallel-ad-1.12.7-c6AdEX -e 'using SafeTestsets; @safetestset "Parallel AD single-thread entry" begin; include("lib/CorleoneBase/test/ad/parallel.jl"); end' > /tmp/opencode/mooncake-single-safe-1.12.log 2>&1
```

Julia 1.10 and 1.11 SafeTestsets isolation each passed **11,632/11,632**: 11,616 backend
contracts plus 16 driver checks (thread/worker availability, actual execution
locations, backend availability, upstream extension loading on every process,
and worker cleanup). Julia 1.10 observed thread IDs 1–4 on driver PID 54334 and
remote OS PIDs 54425, 54429. Julia 1.11 observed thread IDs 1–4 on driver PID
53515 and remote OS PIDs 53706, 53710.

Julia 1.12 single-thread entry passed the parent's one child-success assertion.
The four-thread child passed all 11,616 backend contracts and its driver checks,
including extension loading and cleanup. It observed thread IDs 2–5 on driver
PID 53687 and remote OS PIDs 53769, 53773. Threaded/distributed coverage was not
replaced by serial work or skipped by the single-thread parent.

The first regression-verifier implementation ran AD before Core: Julia 1.12
sequential AD passed 2,143 checks, but Core's existing extension-unloaded
assertion correctly failed after Zygote had been loaded. The verifier now runs
Core first, then explicitly loads the new Mooncake rule before sequential AD.
No Core assertion was changed. The corrected 1.12 rerun reuses its scratch
manifest:

```sh
julia +1.12 --startup-file=no --threads=4 --project=/tmp/opencode/ad-regressions-1.12.7-hbcxnQ lib/CorleoneBase/test/ad/verify_regressions.jl resolved > /tmp/opencode/mooncake-regressions-1.12-final.log 2>&1
```

Final regression observations:

| Julia | Environment | Core (including source/extension assertions) | Sequential AD |
| --- | --- | --- | --- |
| 1.10.12 | `/tmp/opencode/ad-regressions-1.10.12-GzDBP7` | 332 sequential + 20 parallel passed | 2,143 passed |
| 1.11.7 | `/tmp/opencode/ad-regressions-1.11.7-L6HTKj` | 332 sequential + 20 parallel passed | 2,143 passed |
| 1.12.7 | `/tmp/opencode/ad-regressions-1.12.7-hbcxnQ` | 332 sequential + 20 parallel passed | 2,143 passed |

These scratch environments additionally declare ArrayInterface 7.30.2 for Core;
the AD declaration file in the checkout is unchanged. Core runs before loading
the new Mooncake extension, preserving its existing extension-unloaded check;
sequential AD runs with that extension loaded and keeps all six backends and
the supplemental Zygote rule contracts.

The real registered AD group (including SciMLTesting's environment activation
and both SafeTestsets registrations) is also exercised on Julia 1.12:

```sh
JULIA_PKG_PRECOMPILE_AUTO=0 CORLEONE_TEST_GROUP=AD julia +1.12 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.12.7-c6AdEX -e 'include("lib/CorleoneBase/test/runtests.jl")' > /tmp/opencode/mooncake-registered-ad-1.12.log 2>&1
```

Passed: **13,775/13,775** (2,143 sequential AD + 11,632 parallel AD), with no
skips, broken assertions, failures, or errors. SciMLTesting activated
`/tmp/jl_sciml_group_env_D0mJH3`, resolving the same relevant dependency versions
recorded above against this checkout. Both suites ran under their actual
SafeTestsets registrations. Parallel probes observed thread IDs 2–5 on driver
PID 59020 and remote OS PIDs 63359, 63363; extension-loading and post-`rmprocs`
cleanup assertions passed.

### Final Mooncake scope

AutoMooncake reverse mode is verified through the existing shared parallel
harness on Julia 1.10.12, 1.11.7, and 1.12.7, across all 24 combinations and all
existing contracts at unchanged tolerances. ForwardDiff, ReverseDiff, Zygote,
sequential AD, and Core results are preserved. The explicit solve-boundary rule
runs genuine requested-mode parallel primals and differentiates real trajectory
solves serially; parallel reverse-pass scheduling is not claimed. It requires
explicit rule loading and `safetycopy = false`, and is limited to deterministic,
mode-independent, default-output/default-reduction ensembles with read-only
hooks. Stochastic/custom-reduction ensembles, GPUs, external clusters, higher
derivatives, and automatic package extension registration are not certified
here. There are no remaining failures in the requested matrix or entry points.
Changes are confined to tests, test-local documentation/results, and the new
extension file; package metadata and `src` are unchanged. Unrelated user files
were left untouched.

## Mooncake forward-mode phase

This phase extends the shared parallel harness to `AutoMooncakeForward`
alongside the existing registrations, without a separate suite. Changed files:
`parallel.jl` (registration), `parallel_suite.jl` (docstring),
`CorleoneBaseMooncakeExtension.jl` (forward-mode rule), `README.md`, and this
document. No `src` or package-metadata edit was made; pre-existing unrelated
changes were left untouched.

Mooncake marks a `MinimalCtx` rule primitive in `DefaultCtx` too, so the
reverse-mode `@is_primitive` declaration already makes the ensemble
`Core.kwcall` a forward-mode primitive. Forward-mode differentiation therefore
reached the signature and failed with
`MethodError: no method matching frule!!(::Mooncake.Dual{typeof(Core.kwcall),
Mooncake.NoTangent}, ::Mooncake.Dual{@NamedTuple{...}}, ...)`, retained in
`autofwd-probe-rw` runs (the focused reproducer below shows it). The fix is one
`frule!!` in the same extension file taking `Dual` arguments: it runs the
genuine requested-mode primal ensemble (so threaded and distributed execution is
real and the execution-location probes observe it), then differentiates the
actual deterministic serial replay with the incoming Duals via
`Mooncake.build_frule`, returning the replay's tangent with the genuine primal.
The tangent is already the full forward derivative, so no pullback is needed.
The rule rejects custom output/reduction, RNG/seed, and batching keywords and
requires `safetycopy = false`, exactly like the reverse-mode rule.

Focused reproducers (repository root, any resolved Mooncake environment):

```julia
# Before the frule: MethodError: no method matching frule!!(Dual{Core.kwcall}, ...)
using Mooncake, SciMLSensitivity, DifferentiationInterface, SciMLBase, CorleoneBase,
    CommonSolve, OrdinaryDiffEqTsit5
include("lib/CorleoneBase/ext/CorleoneBaseMooncakeExtension.jl")
include("lib/CorleoneBase/test/ad/parallel_suite.jl")
P = ParallelADTests
f = x -> P.ensemble_loss(P.LinearCase{false}(), x, false, EnsembleSerial();
    solve_kwargs = (; safetycopy = false))
value_and_gradient(f, AutoMooncakeForward(), [1.2, 0.4])
```

After the frule, the same probe returns value `1.7552838932325958` and gradient
`[3.0691566576013765, -4.904742426995389]` — the value matches the reverse-mode
Mooncake result exactly and the gradient agrees to ~4e-13. A wrapped
`SequentialProblem` template under `EnsembleThreads` agrees with ForwardDiff to
~4e-12 (value `1.9672118549776847`). A non-fixture default-hook probe
(`/tmp/opencode/autofwd-template-dependencies.jl`, identical to the reverse-mode
one but calling `value_and_gradient(loss, AutoMooncakeForward(), x)`) returned
`2.2928276967982333` and gradient
`[5.44454233113096, 3.3711137426881344, 3.872378809023712]`, agreeing with
ForwardDiff to ~1e-13, confirming dependence on the template's state and
parameters rather than only the fixture's captured trajectory hook.

Full-driver commands reuse the resolved manifests:

```sh
julia +1.12 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.12.7-c6AdEX lib/CorleoneBase/test/ad/parallel.jl > /tmp/opencode/autofwd-full-1.12.log 2>&1
julia +1.10 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.10.12-Vg4zko lib/CorleoneBase/test/ad/parallel.jl > /tmp/opencode/autofwd-full-1.10.log 2>&1
julia +1.11.7~x64 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.11.7-gO7XfS lib/CorleoneBase/test/ad/parallel.jl > /tmp/opencode/autofwd-full-1.11.log 2>&1
```

All three exited successfully. Each run observed real execution locations:
serial work on the driver process, threaded work across multiple default-pool
thread IDs, and distributed work on both remote worker OS PIDs. The 1.12 run
logged `EnsembleThreads execution locations` across thread IDs 2–5 on driver PID
95852 and `EnsembleDistributed execution locations` on PIDs 96017 and 96021.

| Julia | AutoMooncakeForward | Preserved backends |
| --- | --- | --- |
| 1.12.7 | 3,168/3,168 (4m53.4s) | ForwardDiff 3,168; ReverseDiff 2,112; Zygote 3,168; Mooncake 3,168 |
| 1.10.12 | 3,168/3,168 (4m52.1s) | ForwardDiff 3,168; ReverseDiff 2,112; Zygote 3,168; Mooncake 3,168 |
| 1.11.7 | 3,168/3,168 (4m16.9s) | ForwardDiff 3,168; ReverseDiff 2,112; Zygote 3,168; Mooncake 3,168 |

Each AutoMooncakeForward run covers all 24 combinations (three ensemble modes ×
two template kinds × two models × two dynamics forms), three distinct
trajectories and two inputs each, and every analytic primal, prepared/unprepared
gradient, changed-input, repeated-evaluation, and input-preservation assertion
at unchanged tolerances. DI derives the forward-mode gradient per input
coordinate from pushforwards, so each gradient evaluation runs the traced solve
once per coordinate; the frule's primal and serial replay run untraced except
for the replay's tangent propagation. No parallel assertion is skipped or marked
broken, and no tolerance was weakened.

Sequential AD is unchanged with the updated extension loaded (all six backends
at 344 assertions each on Julia 1.12.7, including sequential AutoMooncakeForward
at 1m55.3s; log `autofwd-seqad-1.12.log`; the same six at 344 each on
1.10.12 (`autofwd-seqad-1.10.log`) and 1.11.7 (`autofwd-seqad-1.11.log`)):

```sh
julia +1.12 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.12.7-c6AdEX -e 'Base.include(Main, "lib/CorleoneBase/ext/CorleoneBaseMooncakeExtension.jl"); include("lib/CorleoneBase/test/ad/runtests.jl")' > /tmp/opencode/autofwd-seqad-1.12.log 2>&1
julia +1.10 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.10.12-Vg4zko -e 'Base.include(Main, "lib/CorleoneBase/ext/CorleoneBaseMooncakeExtension.jl"); include("lib/CorleoneBase/test/ad/runtests.jl")' > /tmp/opencode/autofwd-seqad-1.10.log 2>&1
julia +1.11.7~x64 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.11.7-gO7XfS -e 'Base.include(Main, "lib/CorleoneBase/ext/CorleoneBaseMooncakeExtension.jl"); include("lib/CorleoneBase/test/ad/runtests.jl")' > /tmp/opencode/autofwd-seqad-1.11.log 2>&1
```

SafeTestsets isolation and the single-thread entry point use the same commands
as the reverse-mode phase:

```sh
julia +1.10 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.10.12-Vg4zko -e 'using SafeTestsets; @safetestset "Parallel AD isolation" begin; include("lib/CorleoneBase/test/ad/parallel.jl"); end' > /tmp/opencode/autofwd-safe-1.10.log 2>&1
julia +1.11.7~x64 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.11.7-gO7XfS -e 'using SafeTestsets; @safetestset "Parallel AD isolation" begin; include("lib/CorleoneBase/test/ad/parallel.jl"); end' > /tmp/opencode/autofwd-safe-1.11.log 2>&1
julia +1.12 --startup-file=no --threads=1 --project=/tmp/opencode/parallel-ad-1.12.7-c6AdEX -e 'using SafeTestsets; @safetestset "Parallel AD single-thread entry" begin; include("lib/CorleoneBase/test/ad/parallel.jl"); end' > /tmp/opencode/autofwd-single-safe-1.12.log 2>&1
```

Julia 1.10 and 1.11 SafeTestsets isolation each passed **14,801/14,801**: 14,784
backend contracts (3,168 ForwardDiff + 2,112 ReverseDiff + 3,168 Zygote +
3,168 Mooncake + 3,168 AutoMooncakeForward) plus 17 driver checks (thread/worker
availability, actual execution locations, per-backend availability, upstream
extension loading on every process, and worker cleanup). Julia 1.12
single-thread entry passed the parent's one child-success assertion
(`Parallel AD single-thread entry | 1 1`, 12m08.1s); the four-thread child
passed all five backends (14,801 checks) and its driver checks, observing
default-pool thread IDs 2–5 on driver PID 134659 and remote PIDs 134755/134759
for distributed mode. Threaded/distributed coverage was not replaced by serial
work or skipped by the single-thread parent.

Core is unchanged (no `src` or Core-test change): sequential Core plus 4
parallel construction/interface contracts and 15 parallel supported-version
contracts passed on Julia 1.12.7 with the updated extension present but
unloaded (`autofwd-core-1.12.log`):

```sh
julia --startup-file=no --threads=4 --project=/tmp/opencode/ad-regressions-1.12.7-hbcxnQ -e 'include("lib/CorleoneBase/test/core/sequential.jl"); include("lib/CorleoneBase/test/core/parallel.jl")' > /tmp/opencode/autofwd-core-1.12.log 2>&1
```

The real registered AD group (including SciMLTesting's environment activation and
both SafeTestsets registrations) is exercised on Julia 1.12:

```sh
JULIA_PKG_PRECOMPILE_AUTO=0 CORLEONE_TEST_GROUP=AD julia +1.12 --startup-file=no --threads=4 --project=/tmp/opencode/parallel-ad-1.12.7-c6AdEX -e 'include("lib/CorleoneBase/test/runtests.jl")' > /tmp/opencode/autofwd-registered-ad-1.12.log 2>&1
```

Passed: **Sequential AD 2,143/2,143 (6m13.7s) + Parallel AD 14,801/14,801
(10m20.0s)**, no skips, broken assertions, failures, or errors. The unchanged
sequential driver covers ForwardDiff, ReverseDiff, Zygote, Mooncake reverse/forward,
and FiniteDiff; the parallel driver covers ForwardDiff, ReverseDiff, Zygote,
Mooncake reverse mode, and Mooncake forward mode across the full matrix.

### Final Mooncake forward-mode scope

AutoMooncakeForward is verified through the existing shared parallel harness on
Julia 1.10.12, 1.11.7, and 1.12.7, across all 24 combinations and all existing
contracts at unchanged tolerances, with real threaded and distributed execution
observed by the probes. ForwardDiff, ReverseDiff, Zygote, Mooncake reverse mode,
sequential AD, and Core results are preserved. The forward-mode rule runs the
genuine requested-mode primal and differentiates the actual trajectory solves
serially; parallel forward-pass scheduling is not claimed. It requires explicit
rule loading (the same extension file as reverse mode) and `safetycopy = false`,
and is limited to deterministic, mode-independent, default-output/default-
reduction ensembles with read-only hooks. Stochastic/custom-reduction ensembles,
GPUs, external clusters, higher derivatives, and automatic package extension
registration are not certified here. There are no remaining failures in the
requested matrix or entry points.
