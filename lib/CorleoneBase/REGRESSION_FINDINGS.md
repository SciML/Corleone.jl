# CorleoneBase unpushed-change regression findings

Reproducible evidence for the tests, documentation, and dependent AD rules aligned
with the three unpushed commits below. Earlier findings recorded in this file
concerned a different range; they are not verification results for this checkout.
**Status: verification in progress.** Finding 3 was reproduced on both house
Julia versions, then fixed with the user's explicit approval of the one-line
sized-loop resume correction. The regression remains enabled. The user then
replaced mutating finalization, then confirmed that nonmutating trimming was
intended: `finalize_buffer(it) = it.buffer[1:it.current]`. Tests, documentation,
and the Zygote rule now reflect a separate solved-prefix vector while retaining
the iterator's preallocated capacity. Verification is in progress.

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
