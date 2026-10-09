# CorleoneBase unpushed-change regression findings

Reproducible evidence for the docstrings and regression tests added for the
unpushed commits affecting `lib/CorleoneBase`. Findings 1 and 2 were reproduced
against the checked-out sources, root-caused from runtime evidence, and fixed in
this change. Findings 3 and 4 are recorded unchanged.

## Scope and baseline (recorded at session start)

- HEAD: `367f073f4493fa4f93ff65f61393655fb8de90a8`, plus the documentation and
  test commit `41e167f` and this change's fixes.
- Configured upstream: `origin/CJM/CorleoneBase` = `b2619406a3cf45f871cfb9945e2f882beb4a106b`
- Unpushed commits: `4b1361f` Layout changes, `8f2912c` Add initial
  ParallelProblem API, `a7d8d00` Update methods and type stability, `451600d`
  Format, `367f073` Format.
- `lib/CorleoneBase` files in the unpushed range: `src/CorleoneBase.jl`,
  `src/abstractproblem.jl` → `src/abstractsequential.jl` (rename + changes),
  `src/abstractparallel.jl` (new), `src/parallel.jl` (new),
  `test/core/sequential.jl`.
- Unrelated local work preserved and left untouched: the five modified
  `lib/CorleoneGame/library/**` files, and the untracked `plan.md` and
  `testme.jl`.

Baseline worktrees used for comparison (created with `git worktree add --detach`):

- `/tmp/opencode/cb-baseline` — detached at `b261940` (upstream baseline).
- `/tmp/opencode/cb-head-pristine` — detached at `367f073` (HEAD without any fix).

## Matched environments

Julia 1.12.7 (primary) and Julia 1.10.12 (declared minimum). Head and baseline
scratch environments resolved identical key versions: SciMLBase 3.57.0,
CommonSolve 0.2.14, OrdinaryDiffEqTsit5 2.1.5, ArrayInterface 7.30.2,
BenchmarkTools 1.8.0.

## Changed API contracts and documentation added

- `AbstractSequentialProblem` is now `<: SciMLBase.AbstractSciMLProblem`; a
  `SequentialProblem` can be used as an outer solve/ensemble template.
- `CommonSolve.init` now dispatches on
  `algorithm::SciMLBase.AbstractSciMLAlgorithm`; `nothing` and arbitrary
  solver-state objects are rejected by dispatch (the test fixture gained a
  `NoSolve <: AbstractSciMLAlgorithm`).
- `init`/`solve` keyword partitioning moved from a hard-coded `u0`/`p`/`tspan`
  set to `fieldnames(typeof(problem))` via the generated
  `split_problem_kwargs`/`_split_problem_kwargs`. The routed values are
  forwarded unchanged; representation conversion happens in
  `prepare_stage_problem` when `init` remakes the first stage (this fix;
  converting inside the partition reintroduced Finding 1 at stage 1).
- `prepare_stage_problem(::SciMLBase.AbstractSciMLProblem)` accepts any
  `AbstractSciMLProblem` and applies `ArrayInterface.aos_to_soa` to every
  field, returning in-place problems unchanged and the input itself when no
  field changes; a fallback returns non-SciML stage problems unchanged
  (this fix; see Findings 1 and 2).
- New `AbstractParallelProblem` / `ParallelProblem` ensemble API with
  `get_problem`, `get_problems`, `prob_func`, and an ensemble `CommonSolve.solve`
  that routes `output_func`/`reduction`/`u_init`/`safetycopy` to
  `SciMLBase.EnsembleProblem` and all other keywords to `SciMLBase.solve`.

Docstrings were added to every new/changed symbol above and corrected on `init`,
`SequentialProblem`, `SequentialProblemIterator`, `make_buffer`, the abstract
`get_problem`/`transition`/`terminal`, and the parallel solve method.
`ParallelProblem` is listed in `docs/src/corleonebase_api.md` and a parallel
section is in `docs/src/corleonebase.md`. The `using ArrayInterface` in
`src/CorleoneBase.jl` was restored to `import ArrayInterface` (all uses are
qualified; this is behavior-preserving and required to keep QA's
`no_implicit_imports` green).

## Regression tests

`test/core/sequential.jl`:

- `AbstractSequentialProblem <: SciMLBase.AbstractSciMLProblem`.
- `init(problem, nothing)` / `init(problem, :symbol)` throw `MethodError`.
- Keyword partitioning by problem field names, including `it.solve_kwargs`
  contents and an ODE field (`f`) routed to `remake` rather than `solve`;
  routed values are forwarded unconverted.
- Field-named keywords remake only the first problem (custom fixture).
- `prepare_stage_problem` converts every field of any `AbstractSciMLProblem`
  when the representation must change (fixture state converted through
  `ArrayInterface.aos_to_soa`, other fields preserved by identity), returns
  unchanged-field problems as the same object, and returns in-place problems
  unchanged (the restored contracts behind Findings 1 and 2).

`test/core/parallel.jl` (new `Parallel Core contracts` lane, 20 checks on
1.12): construction/interfaces; default and custom `prob_func` receiving the
`ParallelProblem`, the (copied) template, and an `EnsembleContext`; per-trajectory
`sim_id`; `output_func` routing; ordinary solver keyword forwarding;
reproducibility through `seed`; and the required `trajectories` keyword. On
Julia < 1.12 it asserts the documented `MethodError` instead of skipping.

## Verification results (checked-out sources)

- Core, Julia 1.12.7, `Pkg.test` (also via root `GROUP=CorleoneBase` routing):
  Sequential 332/332, Parallel 20/20 — pass.
- Core, Julia 1.10.12, `Pkg.test`: Sequential 332/332, Parallel 6/6 — pass.
- QA, Julia 1.12.7: 23/23 — pass; Julia 1.10.12: 21/21 — pass.
- Docs, Julia 1.12.7 and 1.10.12, `CORLEONE_TEST_GROUP=Docs`: 112/112 — pass
  on both, including `makedocs(checkdocs = :exports)`, doctests, and the
  executed Literate fishing tutorial.
- Root integration, Julia 1.12.7: root `GROUP=Core` passes 61 checks (local
  controls 20, precompile workload 4, layer interface 8, multiple shooting 29);
  root `GROUP=CorleoneBase` routing passes the 332 sequential + 20 parallel
  sublibrary Core checks.
- AD, Julia 1.12.7, full group: 2143/2143 — pass, no skipped cases. All six
  registered backends pass 344 checks each (ForwardDiff, ReverseDiff, Zygote,
  Mooncake, MooncakeForward, FiniteDiff) plus 19 Zygote solve!-rule checks.
  ReverseDiff's previously failing in-place gradient cases (linear and
  nonlinear, `unprepared` and `prepared`) pass. See Finding 1.

## Finding 1 — ReverseDiff in-place AD regression (functional) — FIXED

Original symptom at HEAD: the canonical AD group aborted at the
`AutoReverseDiff` backend (264 pass / 4 error), with

```
TrackedArrays do not support setindex!
  in dynamics at test/ad/fixtures.jl:17  (du .= -p[1] .* u)
```

in the linear/in-place and nonlinear/in-place `unprepared` and `prepared`
gradient cases. All other backends passed 344 each; upstream `b261940` passed
344 with ReverseDiff.

### Cause (established from the failure stack, not assumed)

The failing call was the stage-1 solve inside `init`
(`src/abstractsequential.jl:166` at HEAD). ReverseDiff traces the loss with
`x::ReverseDiff.TrackedArray`, so the solve-time keywords `u0 = [x[1]]`,
`p = [x[2]]` are `Vector{ReverseDiff.TrackedReal}`. The unpushed commits made
`_split_problem_kwargs` apply `ArrayInterface.aos_to_soa` to every
problem-field keyword, and ArrayInterface's ReverseDiff extension implements

```julia
aos_to_soa(x::AbstractArray{<:ReverseDiff.TrackedReal}) =
    reshape(reduce(vcat, x), size(x))
```

(A `(TrackedArray <: AbstractArray{TrackedReal})` input is rebuilt the same
way). The conversion replaces the state with a `TrackedArray`; the solver then
derives its derivative buffer from that representation, and the user's
in-place dynamics `du .= -p[1] .* u` calls `setindex!` on a `TrackedArray`,
which ReverseDiff forbids. Out-of-place dynamics never write into a buffer, so
only the in-place form fails; the primal cases use plain arrays and pass.
`prepare_stage_problem`'s generic remake applied the same conversion to stage
problems in `step!` (stages 2+). The upstream ODE-only method skipped in-place
problems and only converted when `aos_to_soa` changed something, which is why
the baseline passed. The ArrayInterface ReverseDiff extension is the only
converting `aos_to_soa` method in the resolved environments; ForwardDiff,
Zygote, Mooncake, and FiniteDiff values pass through the default identity.

The Zygote solve!-rule errors (19th check of the group) were a second, masked
regression of the same range: the unpushed commits replaced the untyped
`prepare_stage_problem(problem) = problem` fallback with a
`::SciMLBase.AbstractSciMLProblem` method, breaking the Zygote extension's
traced re-implementation of `step!` for its non-SciML `StageProblem` fixtures.
The AD group never reached `zygote_rules.jl` before because the ReverseDiff
backend aborted first; pristine HEAD `367f073` reproduces the same error.

### Fix

- `prepare_stage_problem` returns in-place problems unchanged, queried through
  `SciMLBase.isinplace(problem) === true` (the trait's empty-body fallback
  returns `nothing`, so unimplemented traits are treated as out-of-place), and
  the untyped fallback `prepare_stage_problem(problem) = problem` is restored.
- `split_problem_kwargs` only routes keywords by field names; `init` remakes
  the first stage with the raw routed values and normalizes the result with
  `prepare_stage_problem(remake(prob; initial_kwargs...))`, the baseline
  composition.

Out-of-place conversion is preserved: a `Vector{TrackedReal}` state is still
normalized to a `TrackedArray` before solving (verified directly), which is
what keeps ReverseDiff's tracked states and broadcast derivatives in one array
representation.

## Finding 2 — `prepare_stage_problem` always remakes (repeatable allocation) — FIXED

The new `_prepare_problem` remade every problem even when `aos_to_soa` changed
nothing. `@benchmark CorleoneBase.prepare_stage_problem(template)` on an
out-of-place `ODEProblem` whose fields need no conversion, three fresh
processes each, warmed, matched envs:

| Source | min (ns) | memory (B) | allocs |
| --- | --- | --- | --- |
| HEAD before fix | 5.6–6.1 | 48 | 1 |
| fixed (this change) | 1.8–2.0 | 0 | 0 |
| baseline `b261940` | 1.779 | 0 | 0 |

Fix: `_prepare_problem` applies `aos_to_soa` per field and returns the input
itself when every converted field is identical (`===`) to the original,
remaking only when some field changed. Zero allocations every run, matching
the baseline latency. Unchanged-object reuse is safe here: all fields are
`===`, so the problem is observationally identical, and the docstring and
tests document the contract. Net `solve` allocations did not regress before
the fix (the type-stability change saved more than this cost; see Finding 4).

## Finding 3 — Parallel API is unsupported on Julia 1.10/1.11

Unchanged and out of scope. `abstractparallel.jl` binds the ensemble hook with
`Base.Fix1(prob_func, problem)`; SciMLBase calls that hook with two arguments
(`prob, ctx`). `Base.Fix1` only forwards multiple remaining arguments on
Julia >= 1.12, so on the declared minimum:

```
MethodError: no method matching (::Base.Fix1{typeof(CorleoneBase.prob_func), …})(::SequentialProblem, ::EnsembleContext)
Closest candidates are: (::Base.Fix1)(::Any)
```

The parallel Core lane asserts this `MethodError` on Julia < 1.12, so the
declared 1.10 minimum stays green while the limitation is visible. The parallel
workflow itself is exercised only on Julia >= 1.12. Not fixed here.

## Finding 4 — Type inference (improved, no new instabilities)

`Base.return_types` on function barriers plus `Test.@inferred`, matched envs.
Re-run after the Findings 1–2 fix; all rows concrete and `@inferred`-passing:

| Path | fixed HEAD | baseline `b261940` |
| --- | --- | --- |
| `init(seq3, Tsit5; kwargs)` | concrete, `@inferred` OK | non-concrete `where {…}`, `@inferred` FAIL |
| `solve(seq1, Tsit5; kwargs)` | concrete, `@inferred` OK | non-concrete `SolutionWrapper`, `@inferred` FAIL |
| `solve(seq3, Tsit5; kwargs)` | concrete, `@inferred` OK | non-concrete `SolutionWrapper`, `@inferred` FAIL |
| `prepare_stage_problem(ode)` | concrete, `@inferred` OK | concrete, `@inferred` OK |
| `split_problem_kwargs(ode, nt)` (new) | concrete, `@inferred` OK | n/a |
| `_prepare_problem(ode)` (new) | concrete, `@inferred` OK | n/a |
| parallel `solve(par, Tsit5, EnsembleSerial; trajectories)` (new) | concrete, `@inferred` OK | n/a |

Warmed runtime (allocation) comparison over three fresh processes, `min`
(median) µs and allocation count — unchanged by this fix and no regression on
the solve/init paths:

| Path | fixed HEAD | baseline |
| --- | --- | --- |
| `solve(seq1, …)` | 56.6–60.0 (64.3–67.6) µs, 9809 allocs | 59.6–61.8 (67.6–70.6) µs, 9837 allocs |
| `solve(seq3, …)` | 122.5–123.8 (130.0–133.4) µs, 19167 allocs | 124.2–130.2 (131.5–139.3) µs, 19188 allocs |
| `init(seq3, …)` | 61.0–62.9 (65.4–66.1) µs, 9810 allocs | 62.3–66.2 (67.7–70.8) µs, 9831 allocs |

The solve/init differences are small, consistently in HEAD's favor, and
consistent with the type-stability commit; they are reported as no regression
rather than as a speedup claim.

## New APIs without a comparable baseline

`ParallelProblem` and its `solve`, `AbstractParallelProblem`, `prob_func`,
`get_problems`, `split_problem_kwargs`, `_split_problem_kwargs`, and
`_prepare_problem` do not exist at `b261940`, so no head-vs-baseline runtime or
allocation comparison applies. Their inference is assessed above; only the
`parallel_solve` path is measured for correctness, not compared for performance.

## Coverage limitations

- Parallel behavior is verified only on Julia >= 1.12; on < 1.12 only the
  documented `MethodError` is asserted (Finding 3).
- Performance is measured on a shared workstation with BenchmarkTools' warmup
  and sampling; contention was minimised but not eliminated, so min-of-samples
  is used and only repeatable, code-explained differences are called findings.
- Inference is assessed on representative concrete call sites; this is not a
  claim that every possible usage is type-stable.
- Compiled ReverseDiff tapes are outside the AD suite's scope (unchanged).

## Reproducible commands

Core (1.12 / 1.10), QA, Docs via `CORLEONE_TEST_GROUP`:

```sh
CORLEONE_TEST_GROUP=Core JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no -e 'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"])'
```

Full AD group (all registered backends):

```sh
CORLEONE_TEST_GROUP=AD JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no -e 'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"])'
```

Focused ReverseDiff comparison: recreate the comparison checkouts with

```sh
git worktree add --detach /tmp/opencode/cb-baseline b2619406a3cf45f871cfb9945e2f882beb4a106b
git worktree add --detach /tmp/opencode/cb-head-pristine 367f073f4493fa4f93ff65f61393655fb8de90a8
```

then `Pkg.develop` each `lib/CorleoneBase` checkout into an AD environment and run

```sh
julia --startup-file=no --project=/tmp/opencode/cb-ad /tmp/opencode/ad_reversediff.jl
```

(`test_sequential_ad(AutoReverseDiff(); solve_kwargs = (; sensealg =
SciMLSensitivity.SensitivityADPassThrough()))` against the checked-out
sources.) Inference and performance scripts and raw logs:
`/tmp/opencode/inference.jl`, `/tmp/opencode/inference_fixed.txt`,
`/tmp/opencode/bench.jl`, per-run `RESULT` lines quoted above.
