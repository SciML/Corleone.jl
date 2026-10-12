# Parallel ForwardDiff, ReverseDiff, and Zygote verification

Commands run from `/home/alcap/Projects/Corleone.jl.worktrees/CJM-CorleoneBase`
on Linux x86_64, against the checked-out `lib/CorleoneBase` (0.1.0).
Only test files and the `CorleoneBaseZygoteExtension` were changed; no `src`
or package-metadata change was needed. Pre-existing unrelated changes were left
untouched.

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

## Scope

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
