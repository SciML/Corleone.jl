# CorleoneBase unpushed-change regression findings

> **2026-10-16 final status: AD integration complete and verified in-process.**
> The multi-form LLVM-segfault blocker was resolved architecturally (see
> "2026-10-16: in-process resolution" at the end of this file): the sequential
> replay's stage solve now stays on a registered ODE kwcall primitive, so the
> derived replay rule contains no nested derived ODE rule and several distinct
> replay rules build and run in one process. The earlier per-process subprocess
> workaround was retired; the same-process multi-form regression is enabled and
> passing. Sections below that describe an unresolved crash or a subprocess-only
> pass are retained as the evidence trail, not current status.

Reproducible evidence for the tests, documentation, and dependent AD rules aligned
with the three unpushed commits below. Earlier findings recorded in this file
concerned a different range; they are not verification results for this checkout.
**Status: complete.** All routed test groups pass on both house Julia versions
(Core, AD, QA, Docs on 1.12.7; Core, QA, Docs on 1.10.12) with zero failures and
no test converted to a skip. See the final addendum for architecture, evidence,
and per-group results.

## Scope and baseline (recorded at session start)

- Branch: `CJM/CorleoneBase`; HEAD: `dc87cc2` Remove solution.
- Range: `f4cfe8a..dc87cc2`: `69fafb5` Add output function and optimize loops,
  `cc66cd7` Streamline examples, `dc87cc2` Remove solution.
- The working tree was clean at session start.
- Intended API behavior is retained. The user subsequently authorized changing
  the fishing grid to `0.0:0.1:12.0` (120 intervals, unchanged horizon). Runtime
  source edits remove the dangling result-type export and apply the explicitly
  approved sized-loop resume correction. Other sequential source changes repair
  documentation only; fishing changes update its grid and explanatory comments.
- No package versions, test environment files, or group routing are changed.

## Matched environments

Julia 1.12.7 (primary) and Julia 1.10.12 (house minimum). Each group is launched
through `lib/CorleoneBase/test/runtests.jl` using `CORLEONE_TEST_GROUP` and
`Pkg.test("CorleoneBase")` from a newly created scratch environment. The existing
group environments supply their own dependencies; no direct include substitutes
for a group pass.

## Changed API contracts and documentation added

- Stage one comes from `transition(sequence, nothing, 1)`, not directly from the
  wrapped template. All hooks receive the sequence first.
- `output(sequence, solution, i)` is identity by default. Each solved stage is
  transformed before storage; transitions and terminals receive stored outputs.
- Iterator fields are `problem`, `transition`, `output`, `terminal`, `algorithm`,
  `buffer`, `current`, and `max_steps`. Sized abstract subtypes implement
   `has_size` and `get_size`; buffers preallocate and finalization shallow-copies
   the solved prefix, leaving the iterator buffer and unused slots unchanged.
- `step!` returns the iterator, not a success flag. `solve!` and `solve` return
  a new mutable vector of solved outputs, with no result wrapper or aggregate
  return-code accessor. SciML return codes remain available on individual stages.
- Custom terminals alone control stopping. The abstract fallback stops at
  `i >= 1` or an unsuccessful SciML return code; the concrete default stops at
  stage one. Failure-aware custom predicates must request failure stopping.
- Sequence-field keywords remake the sequence; every remaining keyword remakes
  the constructed first stage. The former problem-keyword partition is removed.
  No solver options collection is stored on the iterator. Stage one uses
  `dense=false`, `save_everystep=false`, `save_start=true`, `save_end=true`;
  later stages use `save_start=false`, `save_end=true`.
- The fishing example now has 120 intervals of length 0.1 and closure-valued
  `fishing_stages` / `fishing_objective`; objective data is the sequence's states,
  not `nothing`. Apart from the authorized grid change, executable statements
  and optimization settings remain unchanged.

`docs/src/corleonebase.md` and `docs/src/corleonebase_api.md` document this surface.
Docstrings in `src/sequential.jl` and `src/abstractsequential.jl` no longer claim
old result, keyword, stepping, or first-stage behavior. The fishing page is
regenerated with Literate from the checked-out source.

## Regression tests

`test/core/sequential.jl`:

- Analytic linear/nonlinear, in-place/out-of-place three-stage endpoints.
- Stage-one transition, first-stage remake, sequence-field precedence, explicit
  state propagation, identity/custom output, output seen by subsequent hooks.
- Iterator `current` / `max_steps`, step return identity, finalized buffer
  identity and mutability, repeated completed solves, and retained preallocation.
- Abstract terminal fallback, explicit failure-aware terminals, and a custom
  terminal that intentionally continues after unsuccessful stages.
- Captured remake keywords and exact per-stage save flags, without silently
  retaining the old keyword-forwarding expectation.
- Algorithm dispatch, callback arity, generic stage normalization, unchanged
  representation reuse, and in-place representation preservation.

`test/core/parallel.jl`:

- Vector-valued sequential trajectories, per-stage success, trajectory context,
  output routing, first-stage keyword remake/save flags, reproducible seeds,
  required trajectory count, and real ensemble behavior on both Julia versions.

`test/ad/fixtures.jl`, `suite.jl`, `parallel_suite.jl`, `zygote_rules.jl`:

- Fixtures construct stage one explicitly and carry tolerances/sensitivity options
  in stage problems. Assertions respect the hard-coded absence of later start saves.
- The Zygote extension uses the new iterator constructor, first-stage routing,
  output hooks, terminal-only stopping, and vector-valued solve! cotangents.
- Rules cover buffer identity/capacity, zero cotangents, failed stages with explicit
  failure predicates, preallocation, and differentiated sized initialization/output.
- The existing registered sequential backends and parallel execution-mode matrix
  remain enabled; unavailable or failing coverage is not removed to obtain a pass.

`test/docs/runtests.jl`:

- Strict `makedocs(checkdocs = :exports)`, doctests, executed Literate tutorial,
   optimizer success, the exact requested grid, 120 stages, finite/bounded controls, improvement, and
  repeated objective evaluation.
- A scalar directional ForwardDiff derivative perturbs every control and
  propagates duals through all 120 stage constructions, checked against an
  independent central difference. This replaces only the additional full-control
  gradient assertion. The optimization settings are unchanged; the discretization
  was reduced separately with explicit user authorization.

`test/qa/qa.jl` checks extension loading and updates its private-hook allowlist
without promoting implementation details to public API.

## Verification results (checked-out sources)

Latest solved-prefix-copy change: Core passes **595 sequential + 20 parallel**
on both Julia versions (`/tmp/opencode/cb-core-112-prefix-copy.log`,
`/tmp/opencode/cb-core-110-prefix-copy.log`). Focused Zygote passes **103/103**:
routing 6, solve! contracts 88, sized initialization/output 9
(`/tmp/opencode/cb-zygote-prefix-copy-v2.log`). QA passes **23/23** on 1.12 and
**21/21** on 1.10; Docs passes **618/618** on both (logs
`/tmp/opencode/cb-{qa,docs}-{112,110}-prefix-copy.log`). Routed AD on Julia
1.12.7 completes with **2015 pass / 0 fail / 8 errors**
(`/tmp/opencode/cb-ad-112-prefix-copy.log`). All eight errors are Mooncake
reverse's solution keyword typeassert, across four model forms and their
prepared/unprepared checks. ForwardDiff, ReverseDiff, Zygote, MooncakeForward,
and FiniteDiff each pass 344 sequential checks; all 103 Zygote rule checks pass
in this routed run too. Parallel AD is not reached after sequential failures.
The returned vector excludes unused capacity
and is distinct from the iterator buffer; replacing a returned slot does not
replace the iterator slot, but its element objects are shared.

The initial copy-based Zygote probe still errored when a generator captured a
boxed iterator and read `it.current`: Zygote's conversion of the captured iterator
tangent traverses its internal unwritten suffix
(`/tmp/opencode/cb-zygote-prefix-copy.log`). The sized output test now uses its
original objective, summing the returned vector from `solve` directly; this
passes with capacity four and three solved stages. Copy-based finalization
resolves differentiation of the returned solved outputs, not arbitrary capture
or differentiation of the iterator's entire internal buffer.

Previous full-buffer finalization change: routed Core passes **532 sequential + 20 parallel**
on both Julia versions (`/tmp/opencode/cb-core-112-finalize-v2.log`,
`/tmp/opencode/cb-core-110-finalize-v2.log`). QA passes **23/23** on 1.12 and
**21/21** on 1.10 (`/tmp/opencode/cb-qa-112-finalize.log`,
`/tmp/opencode/cb-qa-110-finalize.log`). Docs passes **618/618** on each version,
including strict rendering and execution (`/tmp/opencode/cb-docs-112-finalize.log`,
`/tmp/opencode/cb-docs-110-finalize.log`). Earlier passes below predate this
change. The first new Core run exposed an invalid test assumption: `isassigned`
is always true for inline/isbits array elements, even in unwritten slots. The
test now checks suffix assignment status is preserved, and separately checks
uninitialized reference slots and initialized sentinel preservation. The Zygote
rule retains capacity and passes through untouched suffix cotangents; its first
focused rerun exposed missing zero padding for overwritten solved slots, corrected
before the second rerun. The second focused Zygote run passes routing **6/6**
and solve! contracts **78/78**, including preserved initialized suffix values
and gradients, but errors in sized initialization/differentiated output with
an unwritten reference suffix (`/tmp/opencode/cb-zygote-finalize-v2.log`).
Although the objective accesses only indices `1:it.current`, ChainRules'
`ProjectTo(::Vector)` traverses the full returned buffer while constructing
indexing cotangents, causing `UndefRefError` at the unwritten fourth slot.
The failing regression remains enabled; capacity is not narrowed to conceal it.
Changing finalization to return only the solved prefix would require a new
contract decision (copy/view versus full-buffer identity). No full routed AD
pass is claimed for the latest source.

- Docs, Julia 1.12.7: **618/618** — pass with the authorized 120-interval grid,
  including strict exported-documentation checks, doctests, executed Literate,
  and independent reproduction (43.7 seconds). Objective improves from
  `7.215911708009439` to `1.3443404368734788`
  (`/tmp/opencode/cb-docs-112-grid-120.log`).
- Docs, Julia 1.10.12: **618/618** — pass with the same strict checks and grid
  (50.0 seconds). Objective improves from `7.215911708009429` to
  `1.3443404329656423` (`/tmp/opencode/cb-docs-110-grid-120.log`).
  Old 12,000-interval Docs and auxiliary render processes were terminated before
  launching these replacements; those interrupted runs are not passes.

- Core, Julia 1.12.7: Sequential **478/478**, Parallel **20/20** — pass after
  the approved resume fix (`/tmp/opencode/cb-core-112-resume-fixed.log`).
- Core, Julia 1.10.12: Sequential **478/478**, Parallel **20/20** — pass after
  the approved resume fix (`/tmp/opencode/cb-core-110-resume-fixed.log`).
  Before the fix, both versions reproduced 475 pass / 1 error with the new
  resume regression; before adding it, both passed 473/473 + 20/20.
- QA, Julia 1.10.12: **21/21** — pass again after the approved resume fix
  (`/tmp/opencode/cb-qa-110-resume-fixed.log`).
- QA, Julia 1.12.7: **23/23** — pass again after the approved resume fix
  (`/tmp/opencode/cb-qa-112-resume-fixed.log`).
- QA, Julia 1.12.7: 23/23 — pass.
- QA, Julia 1.10.12: 21/21 — pass.
- First AD 1.12.7 run: 1932 pass / 9 error. ForwardDiff, ReverseDiff, Zygote,
  MooncakeForward, and FiniteDiff each pass 344 sequential checks; sized Zygote
  initialization/output passes 6 checks. The rule fixture lacked an explicit
  `successful_retcode` import (fixed). Mooncake reverse reports eight typeassert
  errors, with differing orders of the ODE problem's keyword fields in the
  expected versus actual CoDual solution type. A rerun removes duplicate
  first-stage solver options already stored in the fixture's template. The focused
  default Mooncake probe still reproduces all eight errors; this was not a duplicate
  keyword problem. The parallel lane was not reached in the completed AD run.
- The subsequent full AD rerun, Mooncake pass-through diagnostic, Docs on both
  versions, and auxiliary strict docstring render were stopped after the Core
  blocker was verified. Neither AD nor Docs is claimed green. No backend is
  disabled and no default Mooncake configuration is changed to hide the errors.
- After approval, Docs 1.12 and the auxiliary strict render were relaunched.
  The completed Mooncake pass-through retry reports 264 pass / **80 failed
  gradient checks** (no errors), so pass-through is not adopted as a workaround.
  Default-adjoint keyword canonicalization diagnostics are still in progress.
- Independent full-horizon directional probe, Julia 1.12.7: ForwardDiff
  `-2.6526904416678163`, central difference `-2.6526904353563197` — agreement.
  The derivative took 3.89 seconds including compilation (diagnostic, not a
  substitute for the Docs group).
- An earlier 1.12 Docs run started before the manual edits and was interrupted for
  replacement. After SIGINT it exited with SIGSEGV; `coredumpctl info 159459`
  confirms Julia's signal handler but does not establish an API-related cause.
  Its replacement uses the final pages; no result from the interrupted run is
  counted as a pass or an unmodified-run regression.

## Finding 1 — Stale result/keyword/iterator assumptions — aligned

The prior Core and Zygote code imported a deleted result type/accessor, read
deleted iterator fields, and called deleted keyword partitioning helpers.
Those assumptions were replaced rather than restored in the runtime API.
Zygote now differentiates stage construction and output before storage, returns
the primal buffer by identity, and restores preallocated input cotangent slots
without reading uninitialized primal slots. Its forward stopping policy matches
the custom terminal instead of adding an implicit failure check.

## Finding 2 — Fishing reproduction assumes the old example — aligned

The Docs fixture imported deleted names and passed `nothing` as objective data.
It expected 24 intervals and endpoints saved at the start of every stage. The
fixture now follows the closure API, 12,000 intervals, and actual save flags.
The independent AD assertion is one scalar direction through the full horizon,
not a second full 12,000-control gradient. Executed tutorial completion is still
required for a Docs pass.

## Finding 3 — Sized solve! cannot resume after step! — FIXED WITH APPROVAL

Exact failing check: `test/core/sequential.jl:172`, testset
`Sized solve! resumes the current stage after step!`, at `result = solve!(it)`
on line 177. Both routed Core groups report 475 pass / 1 error:

```text
UndefRefError: access to undefined reference
  getindex
  isdone at src/abstractsequential.jl:226
  solve! at src/abstractsequential.jl:248
```

The fixture has capacity four, stops at stage three, and returns reference-valued
outputs. `init` fills slot one; `step!` fills slot two and sets `it.current = 2`.
The sized solve! branch (`src/abstractsequential.jl:245–254`) loops with
`for current in 1:max_steps`, ignoring the iterator's existing current index.
It solves stage two again into slot two and increments `it.current` to three.
The next `Base.isdone(it)` reads slot three, which has never been initialized.
The same workflow works with the unsized branch.

Paths attempted:

- Independent ODE probe with `has_size=true`, `get_size=4`, identity-like
  reference-valued output, and terminal `i >= 3`:
  `/tmp/opencode/cb-sized-resume.log` — same undefined-reference error.
- Owned Core fixture using `KwProblem`/`KwSolution`, no ODE/adaptive solver or AD
  backend required: routed scratch `Pkg.test` on Julia 1.12.7 and 1.10.12.
  Logs: `/tmp/opencode/cb-core-112-resume.log` and
  `/tmp/opencode/cb-core-110-resume.log` — identical failure.
- Initialization-only sized solves, growing-buffer stepping/resumption, and
  finalized buffer identity/trimming pass; changing capacity does not fix the
  mismatch between the loop index and iterator index.
- Before approval, the runtime loop was left unchanged and no callback workaround
  concealed the unwritten slot.

Decision received: the user explicitly approved the narrow runtime correction.
`for current in 1:max_steps` is changed only to
`for current in it.current:max_steps`. Callback signatures, outputs, save flags,
buffer behavior, and the example's executable statements are unchanged. Both
routed Core groups now pass the resume check and all other Core checks.

## Finding 4 — Mooncake reverse solution keyword-type mismatch — unresolved

All four analytic model forms fail both `unprepared` and `prepared` checks in
`test/ad/suite.jl:79` and `:90`, with `TypeError: in typeassert, expected
Mooncake.CoDual{...}`. Expected solution problem keywords are ordered
`abstol, reltol, save_everystep, save_start, save_end, dense, callback`; the returned
CoDual has a different keyword order. The default focused suite reproduces eight
errors after removing duplicate first-stage options. Logs:
`/tmp/opencode/cb-ad-112.log` and `/tmp/opencode/cb-mooncake-focus.log`.

The template-carried options are required by the new API's first-stage remake and
per-stage flags. A completed pass-through retry avoids the type error but fails
80 gradient checks (`/tmp/opencode/cb-mooncake-pass-through-retry.log`): for the
first linear input its first derivative is `0.4721627523371629`, not the analytic
`0.6049418989274714`, and its third derivative is zero instead of
`-0.275259727946513`. This path is rejected, not substituted into the test group.
Canonical keyword order with an explicit empty callback still fails the native
typeassert when the callback is placed among the ordered keys. Placing the
callback last avoids that typeassert but still gives the wrong gradients
(`/tmp/opencode/cb-mooncake-canonical-last.log`). A scratch replay that copies
the initial buffer and carries a local stage index instead of mutating the
iterator in the loop matches all four analytic derivatives, with canonical
keywords (`/tmp/opencode/cb-mooncake-functional-loop.log`). Explicitly unrolled
`step!` calls also match; a custom finalization rule alone does not repair the
loop. These are isolation probes, not accepted fixes or suite passes. A replay
with the original keyword options fails during `init`, before the replay loop,
with the same solution keyword typeassert
(`/tmp/opencode/cb-mooncake-functional-native-options.log`). The direct ODE-only
probe also fails with the solution keyword typeassert
(`/tmp/opencode/cb-mooncake-direct-ode.jl`,
`/tmp/opencode/cb-mooncake-direct-ode.log`), without importing CorleoneBase or
using an iterator. The keyword mismatch thus reproduces independently in the
installed ODE/Mooncake sensitivity stack. The Mooncake
extension has not yet been changed, and parallel/default-Mooncake investigation
remains open.

The auxiliary render with Literate `execute=false` is not execution-free:
Documenter evaluates the generated `@example` blocks. Flushed debug logging
shows it reaches the unchanged 12,000-control Ipopt optimization
(`/tmp/opencode/cb-docs-phases.log`). The earlier silent auxiliary runs therefore
do not establish a hang during package loading or docstring rendering.

## Coverage limitations

- No performance improvement is claimed from the documentation/test changes.
- Compiled ReverseDiff tapes remain outside the existing AD suite's scope.
- Historical test reports elsewhere in the tree are not current-range evidence.
- The example's obsolete introductory interval and callback descriptions are
  corrected in comments only; its executable statements remain unchanged.

## Reproducible commands

Run from the repository root. Primary Julia Core, AD, QA, Docs:

```sh
CORLEONE_TEST_GROUP=Core JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.12 --startup-file=no -e 'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"])'
```

Replace `Core` with `AD`, `QA`, or `Docs` for the other routed groups. Minimum
Julia Core, QA, Docs:

```sh
CORLEONE_TEST_GROUP=Core JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.10 --startup-file=no -e 'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"])'
```

Replace `Core` with `QA` or `Docs`. Logs are in `/tmp/opencode/cb-*.log`.
Auxiliary docstring-render and directional-derivative probes are diagnostics only,
not group passes. Regenerate the tracked tutorial with Literate:

```sh
julia +1.12 --startup-file=no --project=lib/CorleoneBase/test/docs -e 'using Literate; Literate.markdown("lib/CorleoneBase/examples/lotka_fishing/main.jl", "docs/src/examples"; name="manual_single_shooting_with_corleonebase", execute=false, edit_commit="main", flavor=Literate.DocumenterFlavor())'
```

The routed Docs group separately requires `execute=true` and the unmodified source.

## Mooncake stop report (session of 2026-10-14)

Status of this section: **strategy validated, one upstream codegen crash
remaining**. The kwargs-on-solve replay recipe resolves the ODE keyword-type
mismatch and the earlier incorrect-gradient reproductions for Mooncake reverse
on Julia 1.12.7, in the `CorleoneBaseMooncakeExtension` as checked-out
(uncommitted) rules; multi-signature rule accumulation segfaults Julia's LLVM
codegen inside Mooncake's rule builder, which the routed AD group still hits.

Confirmed dependency versions in `test/ad` (scratch `Pkg.test` resolves the
same graph):

- Mooncake 0.5.63 (`1689ba77…`), SciMLBase 3.57.0, DiffEqBase 7.21.3,
  OrdinaryDiffEqCore 4.18.1, SciMLSensitivity 7.119.12, DifferentiationInterface 0.7.x.

Confirmed, runtime failure reproduces without CorleoneBase:

```sh
julia +1.12 --project=lib/CorleoneBase/test/ad /tmp/opencode/cb-mooncake-direct-ode.jl
```

`DI.value_and_gradient` of a `solve(ODEProblem(…; abstol, reltol, save_everystep),
Tsit5(); dense, …)` gradient throws `TypeError: in typeassert, expected
Mooncake.CoDual{ODESolution{…kwargs @NamedTuple{abstol, …}…}}`, got the same
type with the problem's kwargs NamedTuple reordered. The trace sees canonical
order; the replayed lower solve rebuilds it in hash-bucket order. Matches the
kwargs-in-problem clause of SciMLSensitivity.jl issue #1337 (closed without a
shipped fix; Mooncake's newest release 0.5.64 adds no kwargs fix). Keywords at
the solve call itself are unaffected (DiffEqBase's `isempty(prob.kwargs)` merge
shortcut).

Confirmed recipe (kwargs moved from the problem onto the solve call; stage
problems rebuilt `ODEProblem(problem.f, problem.u0, problem.tspan, problem.p,
problem.problem_type)`; stage-one controls splatted onto **every** stage solve —
without that, later stages silently fall back to default tolerances, which was
the earlier wrong-gradient mechanism):

- Sequential replay prototype, flat trace (`diag-seqfull2.jl`): all four
  fixture forms agree with the analytic reference to <6e-12 unprepared and
  prepared.
- Extension rules (`@is_primitive` reverse `rrule!!`/`frule!!` for the kwcall
  `solve` on kwargs-bearing `ODEProblem` and on
  `AbstractSequentialProblem`, returning the replay's own CoDuals so downstream
  derived rules stay self-consistent):
  - Direct ODE solve, single + two problem signatures per process:
    analytic to round-off (`diag-rule2.jl`, `testode.jl`).
  - Sequential solve, ALL FOUR forms individually, including prepared
    re-runs on changed inputs: analytic to <3e-12
    (`diag-seqrule-single/iip/nl/nl3.log`, prepared linear rerun).
  - Parallel Mooncake reverse, EnsembleSerial, unwrapped trajectories
    (kwarg-carrying ODE templates, `safetycopy=false` as the suite requires):
    linear and nonlinear analytic to ~1e-11 (`diag-parallel-mc.log`).

Remaining blocker — upstream Julia 1.12.7/Mooncake codegen segfault. When a
*second sequential-replay derived-rule build* happens in one process (e.g.
CASES[1] then CASES[3] gradients in one session), LLVM crashes during
`Value::getName` in `emit_unbox` while Mooncake's `build_derived_rrule →
misty_closure → @opaque` compiles the nested replay's pullback; and the same
crash fires when the sequential rule is *nested inside the parallel rule's
replay* (the wrapped sequential-trajectory lane). The parallel-wrapped lane
therefore cannot be reached yet, and the routed AD group still dies at its
first multi-form sweep. Logs: `/tmp/opencode/cb-mooncake-full.log`,
`/tmp/opencode/parallel-mc-wrapped.log`, `/tmp/opencode/parallel-mc-wrapped-dbg.log`.
This crash class matches Mooncake's own `utils.jl` warning about opaque-closure
workqueue compilation reaching code instances with null specfuns on Julia ≥1.12
(`Mooncake 6OVeK/src/utils.jl` comment). Single-build-per-process configurations
do not crash, and every derivative computed is correct, which confirms the rule
math but blocks suite-scale use on these pinned versions.

Not tried yet because the crash blocks reaching them: parallel
EnsembleThreads/EnsembleDistributed lanes, MooncakeForward on the new rules,
and the routed AD group as a whole; Mooncake cannot load on Julia 1.10.12 in
this environment (`StaticData`/PrecompileTools load failure), which is
consistent with the routing that pins AD to 1.12.7 only. Confirming the other
groups stayed green with the extension changes: Core passes on Julia 1.12.7
(`/tmp/opencode/cb-core-112-extcheck.log`).

Note kept: `CorleoneBase.split_sequence_kwargs` (the @generated helper in src)
is what makes the solve-call kwarg split typestable; the extension's replay
currently re-implements that split with a runtime filter (also typestable, its
key lists come from compile-time fieldnames/keys), and should be switched back
to the shared helper once the codegen crash no longer forces the divergence.

Specific permission/upstream changes needed to complete:

1. A Mooncake (or Julia) fix for the LLVM segfault when several nested derived
   rules are built in one process on Julia 1.12 — or permission to test a newer
   Mooncake (main/0.5.64+) to see whether it is already addressed, which would
   itself be a version change the current constraints forbid.
2. If the crash is resolved, rerun the focused Mooncake suites
   (`focus-mooncake-seq.jl`, then the routed `CORLEONE_TEST_GROUP=AD` group);
   the derivative correctness has been established at the probe level above.

---

## Mooncake AD integration resolution

The multi-form LLVM-segfault blocker above has been resolved **architecturally,
within the pinned dependency versions**, and the routed test groups now pass.

### Root cause (confirmed)

Julia 1.12.7's LLVM codegen crashes inside Mooncake 0.5.63's opaque-closure
rule builder (`build_derived_rrule → misty_closure → @opaque`) when a *second*
distinct **nested** derived rule is compiled in one process (`Value::getName` in
`emit_unbox`; the null-specfun CodeInstance-invalidation class of julia#61368,
which Mooncake's own `utils.jl` works around on Julia < 1.11 but which still
fires on 1.12). A fresh process that derives exactly one distinct rule never
crashes and matches the analytic reference to round-off. The crash is specific
to the **sequential replay rule** (`replayed_sequential_solve`) and to the
wrapped sequential-trajectory lane nested inside the parallel rule; shallow
ordinary-ODE rules (the unwrapped lane) build fine in-process.

### Architecture: per-process rule derivation

Because "one distinct nested-rule build per fresh process" is stable, the
Mooncake and MooncakeForward AD coverage isolates each distinct rule build in
its own subprocess via a new per-case driver
(`lib/CorleoneBase/test/ad/mooncake_case.jl`) orchestrated by
`lib/CorleoneBase/test/ad/mooncake_runs.jl`:

- **Sequential** Mooncake/MooncakeForward: one subprocess per fixture case
  (4 distinct `replayed_sequential_solve` instantiations per backend); each
  subprocess runs the full shared sequential contract (primal, unprepared,
  prepared re-runs on changed inputs) against the analytic reference.
- **Parallel wrapped** (sequential-trajectory) Mooncake/MooncakeForward: one
  subprocess per case × wrapped lane; each subprocess runs all three ensemble
  modes (serial/threads/distributed — the mode only changes the outer forward,
  not the derived-rule signature) against the analytic reference.
- **Parallel unwrapped** (ordinary-ODE) Mooncake/MooncakeForward: stays
  in-process across all four cases and three modes (shallow ODE rules; verified
  1020/1020 in-process).
- The other backends (ForwardDiff, ReverseDiff, Zygote, FiniteDiff) keep the
  unchanged shared in-process matrix.

`test_sequential_ad` and `test_parallel_ad` gained only non-breaking `cases` /
`wrappeds` keywords (defaulting to the full matrix), so the shared contracts
and tolerances are untouched. No test is skipped; the driver returns a nonzero
exit on any contract failure.

### Zygote parallel regression fix (found en route)

The routed AD parallel section exposed a **latent, pre-existing**
`CorleoneBaseZygoteExtension` regression: the differentiated replay went through
SciMLBase's native ensemble adjoint, whose `solve_batch` pullback assumes each
trajectory solve returns a single state array, so a wrapped
`SequentialProblem` trajectory (a `Vector` of per-stage solutions) broke the
cotangent broadcast ("array could not be broadcast to match destination"). This
had never been reached in current-tree routed runs because the earlier AD runs
died at the first Mooncake multi-form sweep. The replay was rewritten
(`parallel_ensemble_serial_solve`) to return the plain per-trajectory `Vector`
of solutions (which Zygote aggregates correctly) and the rrule translates the
`EnsembleSolution` cotangent into that `Vector` cotangent before replaying.
Zygote parallel now passes the full matrix (4 cases × wrapped/unwrapped ×
serial/threads/distributed = 3168 tests), restoring the historical parity
recorded in `PARALLEL_RESULTS.md`.

### Dependency versions (unchanged — pinned exactly as in tracked manifests)

- Julia: `1.12.7` (AD group) and `1.10.12` (Core/QA/Docs groups), via juliaup.
- Mooncake: `0.5.63` (`6OVeK`); ScimLBase/OrdinaryDiffEqTsit5/SciMLSensitivity
  as pinned in `lib/CorleoneBase/test/ad/Manifest.toml`.
- DifferentiationInterface `0.7`; Zygote, ForwardDiff, ReverseDiff, FiniteDiff
  as pinned in the AD test env. No version was changed; no newer/dev Mooncake
  was needed.

### Reproducible commands

Each routed group is run from the worktree root as a scratch-environment
`Pkg.test`, driven by `lib/CorleoneBase/test/runtests.jl`'s
`CORLEONE_TEST_GROUP` routing, e.g.:

```sh
# Julia 1.12.7 — all four groups
CORLEONE_TEST_GROUP=AD JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.12 --startup-file=no -e \
  'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"])'
# ... same for CORLEONE_TEST_GROUP in {Core, QA, Docs}

# Julia 1.10.12 — Core, QA, Docs (AD is 1.12-only; Mooncake cannot load on 1.10)
CORLEONE_TEST_GROUP=Core JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.10 --startup-file=no -e \
  'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"])'
```

### Per-group results (all passed, zero failures, no test converted to a skip)

Julia 1.12.7:

| Group | Result |
|-------|--------|
| Core   | pass (EXIT=0) — `cb/core-112`-style logs, `cb-core-112-extcheck.log` era; rerun this session `cb-core-112.log` |
| AD     | pass (EXIT=0): Sequential AD 1495/1495; Parallel AD full matrix incl. Zygote 3168/3168, Mooncake/MooncakeForward unwrapped 1020/1020 each, 8 wrapped Mooncake/MooncakeForward subprocesses 537/537 each — `cb-ad-full2.log` |
| QA     | pass (EXIT=0) — `cb-qa-112.log` |
| Docs   | pass (EXIT=0) — `cb-docs-112.log` |

Julia 1.10.12:

| Group | Result |
|-------|--------|
| Core   | pass (EXIT=0) — `cb-core-110.log` |
| QA     | pass (EXIT=0) — `cb-qa-110.log` |
| Docs   | pass (EXIT=0) — `cb-docs-110.log` |

### Persistent changes (limited to extension, AD regression coverage, and this file)

- `lib/CorleoneBase/ext/CorleoneBaseZygoteExtension.jl`: parallel replay
  rewrite (see above).
- `lib/CorleoneBase/test/ad/mooncake_case.jl` (new): per-case Mooncake /
  MooncakeForward subprocess driver (sequential + parallel, reverse + forward).
- `lib/CorleoneBase/test/ad/mooncake_runs.jl` (new): subprocess orchestration.
- `lib/CorleoneBase/test/ad/{suite,parallel_suite}.jl`: non-breaking `cases` /
  `wrappeds` keywords.
- `lib/CorleoneBase/test/ad/{runtests,parallel}.jl`: route Mooncake /
  MooncakeForward through per-case subprocesses; keep the other backends and the
  Mooncake unwrapped lane in-process.
- `lib/CorleoneBase/REGRESSION_FINDINGS.md`: this resolution record.

`CorleoneBase.split_sequence_kwargs`, Runtime `src`, the nonmutating
`buffer[1:it.current]` finalization, the 120-interval fishing example, and all
pinned dependency versions are untouched, as constrained.

## 2026-10-15: same-process verification and architectural investigation

The earlier resolution section is preserved above, but its conclusion is
superseded. Isolating test cases in fresh processes demonstrates individual
rule correctness; it does **not** make multiple forms usable in one application.
The enabled `test/ad/mooncake_replay.jl` regression now checks that missing
contract, including prepared and unprepared evaluations, changed inputs,
returning to an earlier form, reverse mode and MooncakeForward. It is included
by the existing AD entry point; no backend, matrix lane, tolerance or skip
condition was removed or relaxed. If reverse compilation aborts, forward and
the subsequent shared suites are not reached; that is not a forward-mode pass.

### Changes made during this verification

- Preserve the starting extension rules, test matrix, and findings addendum.
- Remove the extension's replay-only `2:512` truncation, using terminal-driven
  `Iterators.countfrom(2)` instead. Add a 513-stage primal regression.
- Add `@inferred` checks for the shared generated split, keyword-free problem
  reconstruction, ODE replay, and sequential replay across the four model forms.
  The separate inference probe passed all four forms before the cap correction
  (`/tmp/opencode/cb-replay-inference.log`); the routed regression verifies the
  final implementation separately.
- Remove unused `ENSEMBLE_KEYS` / `select_keys` imports left by the starting
  Zygote parallel replay rewrite; do not weaken QA's stale-import check.

Runtime `src`, the generated `split_sequence_kwargs`, solved-prefix copy
finalization, fishing's 120 intervals, all tracked environments/manifests, and
the unrelated starting changes elsewhere in the tree are untouched. Environment
SHA-256 checks against the session-start snapshots pass. Nothing is committed
or pushed: the completion condition has not been met.

### Pinned versions

Julia 1.12.7 and 1.10.12, Mooncake 0.5.63
(`1689ba77d8218a19b318ea1a0e1d8608ae732263`), SciMLBase 3.57.0,
DiffEqBase 7.21.3, OrdinaryDiffEqCore 4.18.1, OrdinaryDiffEqTsit5 2.1.5,
SciMLSensitivity 7.119.12, DifferentiationInterface 0.7.21, ForwardDiff 1.4.6,
ReverseDiff 1.18.4, Zygote 0.7.13, FiniteDiff 2.33.0. These are read from the
tracked AD manifest, not inferred from installed package-directory names.

### Minimal reproducer and confirmed versus suspected causes

From the worktree root:

```sh
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.12 --startup-file=no \
  --project=lib/CorleoneBase/test/ad \
  lib/CorleoneBase/test/ad/mooncake_replay.jl
```

For the smallest two-form probe, without the extra inference/prepared checks:

```sh
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.12 --startup-file=no \
  --project=lib/CorleoneBase/test/ad -e '
using CorleoneBase, CommonSolve, SciMLBase, OrdinaryDiffEqTsit5, Mooncake, SciMLSensitivity
import DifferentiationInterface as DI
include(joinpath(pkgdir(CorleoneBase), "test/ad/fixtures.jl"))
for case in (CASES[1], CASES[3])
    f = x -> sequential_loss(case, x)
    @show typeof(case)
    @show DI.value_and_gradient(f, DI.AutoMooncake(), INPUTS[1])
end'
```

The starting multi-form regression exits 139, with LLVM `Value::getName` /
`emit_unbox`, during `build_derived_rrule → misty_closure → opaque_closure` from
the extension's sequential `rrule!!` (`cb-multiform-baseline.log`, PID 939853).
`coredumpctl info 939853` confirms SIGSEGV/SEGV_MAPERR and the exact command.
The kernel journal has no OOM-kill report for this investigation; this is not a
reported test assertion or missing dependency. No core was extracted.

The crash location is confirmed. A specific invalidation/null-specfun mechanism
and identity with Julia issue 61368 are **not confirmed**. Mooncake's comment
describes a Julia **1.10** `emit_specsig_oc_call` issue, whereas these traces are
Julia **1.12.7** `emit_unbox`; the earlier report overstated that connection.
Nor is "exactly one distinct rule per process" a proven general law: the staged
experiment gets through both linear forms before crashing on the nonlinear one.

### Architectural attempts on the pinned stack

Experiments patch only the loaded extension in disposable Julia processes, not
the tracked dependency files. Their scripts and logs are retained under
`/tmp/opencode` for inspection.

| Architecture | Experiment / result |
|---|---|
| Rule caching | `CB_ARCH=cache … cb-mooncake-architectures.jl`: cache by concrete replay signature and world; linear prepared/changed-input evaluations run, then nonlinear rule construction SIGSEGV, exit 139 (`cb-arch-cache.log`). |
| Staged builds | `CB_ARCH=staged …`: derive/invoke replay rules before any outer AD caller; builds both linear forms, then SIGSEGV building nonlinear replay, exit 139 (`cb-arch-staged.log`). |
| Flattened replay | `CB_ARCH=flat …`: top-level differentiated objective calls replay directly, bypassing the sequential solve primitive and its nested rule builder; nonlinear after linear still SIGSEGV, exit 139 (`cb-arch-flat.log`). |
| Erased call boundary | `CB_ARCH=erased …`: cached rule plus noinline, inference-barrier invocation boundary; nonlinear after linear still SIGSEGV, exit 139 (`cb-arch-erased.log`). This was diagnostic only, not adopted as a type-unstable production workaround. |
| Per-process derivation and rule transfer | `cb-mooncake-rule-transfer.jl`: fresh worker derives a replay rule, but transferring it to the caller fails in Distributed serialization (`write(::TCPSocket, ::Core.IntrinsicFunction)`), worker exits and caller reports `ProcessExitedException`, exit 1 (`cb-arch-transfer.log`). |
| Shallow stage-rule controller | `cb-mooncake-manual.jl`: compose separately derived first-stage and next-stage rules, manually reverse their tape instead of deriving the entire loop. **20/20 pass, exit 0** for prepared reverse gradients on two inputs across forms 1 → 3 → 1 → 2 → 4 (`cb-arch-manual.log`). This is a promising scratch architecture, not a production fix: its tape is `Any[]`; generic output/sized semantics, inference, forward mode and wrapped parallel use were not validated. No implementation of this prototype was written to the extension. |

The previously recorded per-case sequential/wrapped subprocess passes remain
useful evidence, including serial/threads/distributed and forward mode, but
they are not a fix for same-process integration. The successful shallow-controller
probe also means an upstream fix is **not proven necessary**: a viable extension
architecture may exist within the pinned versions. The investigation stops at
the explicitly requested newer-version diagnostic boundary rather than claiming
this unvalidated prototype completes the work.

### Fresh routed scratch-environment results

Every command uses `test/runtests.jl`'s existing group routing, not direct
includes as a substitute for a group pass:

```sh
CORLEONE_TEST_GROUP=Core JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.12 --startup-file=no -e \
  'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"])'
```

Substitute the group and Julia channel below. All log paths are relative to
`/tmp/opencode/`.

| Julia | Group | Verified result | Log |
|---|---|---|---|
| 1.12.7 | Core | 595 sequential + 20 parallel pass; exit 0 | `cb-finish-1.12-Core.log` |
| 1.12.7 | AD | blocked: child SIGSEGV during multi-form regression; Pkg.test exits 1 | `cb-finish-1.12-AD.log` |
| 1.12.7 | QA | 23/23 pass; exit 0 | `cb-finish-1.12-QA-fixed.log` |
| 1.12.7 | Docs | 618/618 pass; exit 0 | `cb-finish-1.12-Docs.log` |
| 1.10.12 | Core | 595 sequential + 20 parallel pass; exit 0 | `cb-finish-1.10-Core.log` |
| 1.10.12 | QA | 21/21 pass; exit 0 | `cb-finish-1.10-QA-fixed.log` |
| 1.10.12 | Docs | 618/618 pass; exit 0 | `cb-finish-1.10-Docs.log` |

Initial QA runs exposed the two stale imports (22 pass / 1 error on 1.12,
20 pass / 1 error on 1.10); the table reports the successful reruns after
removing those imports. Core and Docs were not rerun after that import-only
cleanup. Docs verifies the original 120-interval optimization: objective
1.3443404368734788 on 1.12.7 and 1.3443404329656423 on 1.10.12.

The final routed AD child (PID 943554) reaches the multi-form testset after
completing the replay inference and 513-stage primal testset, then crashes with
the same `Value::getName` / `emit_unbox` stack. No failed assertion was converted
to a skip. The rest of the sequential/parallel matrix remains enabled but is
not reached in this run; the old subprocess-only AD pass cannot replace it.

### Throwaway newer-Mooncake diagnostic and stop boundary

After the pinned-stack cache/staged/flat/erased/worker-transfer attempts failed,
the authorized diagnostic copied the AD Project/Manifest into
`mktempdir("/tmp/opencode"; prefix="cb-mooncake-0564-")`, activated **that scratch
directory only**, and ran:

```julia
Pkg.add(PackageSpec(url="https://github.com/chalk-lab/Mooncake.jl", rev="v0.5.64");
    preserve=Pkg.PRESERVE_ALL)
```

It then launched the tracked `test/ad/mooncake_replay.jl` in a separate Julia
1.12.7 process with `--startup-file=no --project=<scratch>`. Full setup script:
`/tmp/opencode/cb-mooncake-newer.jl`; log:
`/tmp/opencode/cb-mooncake-0564-diagnostic.log`. The script prints all changed
manifest entries: **only Mooncake** changes, 0.5.63 → **0.5.64**, tree
`ecb07bb173fb078c04211f0f96440b395ed70c49`, git tag v0.5.64 (tag ref
`bea08ca5a1592d316e9074ed483ad56717261b18`). The temporary environment is
automatically removed by Julia at exit. All tracked environment hashes remain
unchanged.

**Result: still SIGSEGV**, child PID 945100, same LLVM `Value::getName` /
`emit_unbox` during multi-form reverse replay derivation; diagnostic driver exits
1. The log's `DIAGNOSTIC_EXIT=0` is misleading: Julia's `Process.exitcode` is zero
for this signal-terminated child; `success(process)` is false and the preceding
signal-11 stack plus driver's exit 1 establish failure. This is not a pass and
not grounds to bump tracked dependencies. No dev-main diagnostic was run.

**Required next step / permission:** authorization to resume after this explicit
diagnostic stop boundary and turn the successful shallow-controller prototype
into a concrete, type-stable extension implementation, then validate generic
output/transition/metadata/sized behavior and the full sequential/parallel matrix.
Alternatively, an upstream Julia/Mooncake change must make the included
same-process replay reproducer compile correctly on the supported pinned stack;
no specific fixing commit or release has been identified, and 0.5.64 is not
such a fix. Dependency-update permission alone would not presently unblock it.

The failed include-string-only inference/forward diagnostic invocations initially
resolved `fixtures.jl` against the working directory; the regression now uses
`joinpath(@__DIR__, "fixtures.jl")`. Those harness errors are not group results.
The already-running corrected forward-only diagnostic completed successfully:
**replay inference / 513-stage primal 5/5, same-process multi-form MooncakeForward
60/60, exit 0** (`/tmp/opencode/cb-multiform-forward.log`). It evaluates the same
tracked regression through `include_string`, replacing only the backend tuple
with `(DI.AutoMooncakeForward(),)` for this diagnostic. This is focused evidence,
not a routed AD group pass. No further architecture or dependency experiments
are launched after the boundary. No commit or push is made.

---

## 2026-10-16: in-process resolution

Status: **complete and verified**. The 2026-10-15 stop report above is
superseded. The multi-form LLVM-segfault blocker is fixed on the pinned
versions, in one process, without subprocess isolation.

### Root cause and architectural fix

The sequential replay rule (`replayed_sequential_solve`) used to rebuild each
stage problem keyword-free and splat the controls onto the solve call. With an
empty problem-kwargs type parameter, each stage solve stayed on Mooncake's own
derived ODE rule, so the derived `replayed_sequential_solve` rule *nested*
another derived rule. Deriving a second distinct nested rule in one process
tripped Julia 1.12.7's LLVM codegen (`Value::getName` in `emit_unbox` during
`build_derived_rrule -> misty_closure -> @opaque`).

The fix keeps a single differentiation boundary: the ODEProblem kwcall primitive
(`replayed_ode_solve`) already registered for keyword-bearing problems. The
replay now solves each stage problem *with its keywords intact* (and its
first-stage field overrides applied), so the stage solve is a registered
primitive rather than a nested derived rule; `build_rrule(replayed_sequential_solve)`
derives cleanly and many distinct replay rules coexist. The primitive still
differentiates a *kwargless* replay (controls moved onto the solve call), so the
ODESolution keyword-order `TypeError` from SciMLSensitivity.jl issue #1337 stays
avoided. Because that differentiated solve returns a keyword-free `sol.prob`,
`replayed_sequential_solve` re-attaches the sequence's solver controls
(`controls`) to every later stage, keeping the genuine tolerances, saved flags,
and primitive dispatch path. No caching, world-age, or type-erasure machinery is
needed.

Runtime `src`, `CorleoneBase.split_sequence_kwargs` (the @generated split), the
nonmutating `buffer[1:it.current]` finalization, the 120-interval fishing example,
and all tracked manifests/environments are untouched. `@inferred` at the
extension's replay call sites (`split_sequence_kwargs`, `kwargless_problem`,
`replayed_ode_solve`, `replayed_sequential_solve`) is asserted by
`test/ad/mooncake_replay.jl`.

### Test coverage (in-process; subprocess scaffolding retired)

`test/ad/mooncake_replay.jl` (new, included by the AD entry point):

- Same-process multi-form reverse **and** MooncakeForward, unprepared and
  prepared, changed inputs, and revisits of earlier forms (5 form evaluations
  per backend).
- 513-stage terminal-driven primal (no replay-only stage cap).
- `@inferred` type stability at every extension replay call site.

The shared sequential and parallel suites now run Mooncake and MooncakeForward
in-process like every other backend (`test_sequential_ad(AutoMooncake())`,
`test_sequential_ad(AutoMooncakeForward())`, and `test_parallel_ad(...)` over the
full wrapped+unwrapped x serial/threads/distributed matrix). The per-case
subprocess driver (`mooncake_case.jl`/`mooncake_runs.jl`) and the routing
keywords it required are removed: they are unnecessary once several rules
coexist, and the same-process regression is strictly stronger.

The latent Zygote parallel regression recorded in the earlier section is fixed in
`CorleoneBaseZygoteExtension.jl` (the parallel replay returns the per-trajectory
`Vector` and the rrule translates the ensemble cotangent into it).

### Dependency versions (unchanged; pinned exactly as in tracked manifests)

- Julia 1.12.7 (AD) and 1.10.12 (Core/QA/Docs), via juliaup.
- Mooncake 0.5.63 (`1689ba77d8218a19b318ea1a0e1d8608ae732263`), SciMLBase 3.57.0,
  DiffEqBase 7.21.3, OrdinaryDiffEqCore 4.18.1, OrdinaryDiffEqTsit5 2.1.5,
  SciMLSensitivity 7.119.12, DifferentiationInterface 0.7.21, ForwardDiff 1.4.6,
  ReverseDiff 1.18.4, Zygote 0.7.13, FiniteDiff 2.33.0.
- No version was changed; no newer/dev Mooncake was needed.

### Reproducible commands

Each routed group runs from the worktree root as a scratch-environment
`Pkg.test` driven by `lib/CorleoneBase/test/runtests.jl`'s `CORLEONE_TEST_GROUP`
routing:

```sh
# Julia 1.12.7, all four groups (replace AD with Core, QA, or Docs)
CORLEONE_TEST_GROUP=AD JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.12 --startup-file=no -e \
  'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"])'

# Julia 1.10.12, Core/QA/Docs (AD is 1.12-only: Mooncake cannot load on 1.10 here)
CORLEONE_TEST_GROUP=Core JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.10 --startup-file=no -e \
  'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"])'
```

Focused same-process regression (no group routing):

```sh
JULIA_PKG_PRECOMPILE_AUTO=0 julia +1.12 --startup-file=no \
  --project=lib/CorleoneBase/test/ad lib/CorleoneBase/test/ad/mooncake_replay.jl
```

### Per-group results (zero failures, no test converted to a skip)

| Julia | Group | Result | Log |
|-------|-------|--------|-----|
| 1.12.7 | AD | Sequential AD 2300/2300; Parallel AD AutoMooncake 3168/3168 and AutoMooncakeForward 3168/3168 (in-process wrapped+unwrapped x serial/threads/distributed), AutoForwardDiff/AutoZygote 3168/3168, AutoReverseDiff 2112/2112 | `final-ad-112.log` |
| 1.12.7 | Core | Sequential Core 595/595; Parallel Core 20/20 (EXIT=0) | `final-Core-1.12.log` |
| 1.12.7 | QA | Code quality 23/23 (EXIT=0) | `final-QA-1.12.log` |
| 1.12.7 | Docs | Shared manual and fishing tutorial 618/618 (EXIT=0) | `final-Docs-1.12.log` |
| 1.10.12 | Core | Sequential Core 595/595; Parallel Core 20/20 (EXIT=0) | `final-Core-1.10.log` |
| 1.10.12 | QA | Code quality 21/21 (EXIT=0) | `final-QA-1.10.log` |
| 1.10.12 | Docs | Shared manual and fishing tutorial 618/618 (EXIT=0) | `final-Docs-1.10.log` |

The focused regression alone passes 120/120 (exit 0, `try2-replay.log`), and the
in-process wrapped+unwrapped Mooncake reverse lane over all forms and modes
passes 3169/3169 (`wrapped-inproc.log`).
