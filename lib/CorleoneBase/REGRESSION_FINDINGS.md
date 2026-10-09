# CorleoneBase unpushed-change regression findings

Reproducible evidence for the docstrings and regression tests added for the
unpushed commits affecting `lib/CorleoneBase`. Runtime implementation was not
changed to fix any finding here.

## Scope and baseline (recorded at session start)

- HEAD: `367f073f4493fa4f93ff65f61393655fb8de90a8`
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
- `/tmp/opencode/cb-head-pristine` — detached at `367f073` (HEAD without the
  documentation/test edits in this change).

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
  `split_problem_kwargs`/`_split_problem_kwargs`, with
  `ArrayInterface.aos_to_soa` applied to the problem-field values.
- `prepare_stage_problem(::SciMLBase.AbstractSciMLProblem)` now always `remake`s
  the problem through the generated `_prepare_problem`, applying `aos_to_soa` to
  every field. The earlier ODE-only method skipped in-place problems and
  returned the argument unchanged when `u0`/`p` were not converted.
- New `AbstractParallelProblem` / `ParallelProblem` ensemble API with
  `get_problem`, `get_problems`, `prob_func`, and an ensemble `CommonSolve.solve`
  that routes `output_func`/`reduction`/`u_init`/`safetycopy` to
  `SciMLBase.EnsembleProblem` and all other keywords to `SciMLBase.solve`.

Docstrings were added to every new/changed symbol above and corrected on `init`,
`SequentialProblem`, `SequentialProblemIterator`, `make_buffer`, the abstract
`get_problem`/`transition`/`terminal`, and the parallel solve method.
`ParallelProblem` is now listed in `docs/src/corleonebase_api.md` and a parallel
section was added to `docs/src/corleonebase.md`. The `using ArrayInterface` in
`src/CorleoneBase.jl` was restored to `import ArrayInterface` (all uses are
qualified; this is behavior-preserving and required to keep QA's
`no_implicit_imports` green).

## Regression tests added

`test/core/sequential.jl` (334 checks with the previous suite):

- `AbstractSequentialProblem <: SciMLBase.AbstractSciMLProblem`.
- `init(problem, nothing)` / `init(problem, :symbol)` throw `MethodError`.
- Keyword partitioning by problem field names, including `it.solve_kwargs`
  contents and an ODE field (`f`) routed to `remake` rather than `solve`.
- Field-named keywords remake only the first problem (custom fixture).
- `prepare_stage_problem` converts every problem field for any
  `AbstractSciMLProblem`, is value-preserving for out-of-place and in-place
  `ODEProblem`s, and always returns a fresh problem (the changed contract
  behind Finding 2).

`test/core/parallel.jl` (new `Parallel Core contracts` lane, 20 checks on
1.12): construction/interfaces; default and custom `prob_func` receiving the
`ParallelProblem`, the (copied) template, and an `EnsembleContext`; per-trajectory
`sim_id`; `output_func` routing; ordinary solver keyword forwarding;
reproducibility through `seed`; and the required `trajectories` keyword. On
Julia < 1.12 it asserts the documented `MethodError` instead of skipping.

## Verification results (checked-out sources)

- Core, Julia 1.12.7, `Pkg.test` (also via root `GROUP=CorleoneBase` routing):
  Sequential 334/334, Parallel 20/20 — pass.
- Core, Julia 1.10.12, `Pkg.test`: Sequential 334/334, Parallel 6/6 — pass.
- QA, Julia 1.12.7, `Pkg.test`: 23/23 — pass. Before the docstring and
  `import ArrayInterface` fixes, QA failed the public-API-docstring check
  (`:ParallelProblem`) and errored `no_implicit_imports` on the `using
  ArrayInterface` introduced by the unpushed commits.
- Docs, Julia 1.12.7 and 1.10.12, `CORLEONE_TEST_GROUP=Docs`: 112/112 — pass
  on both, including `makedocs(checkdocs = :exports)` with the new
  `ParallelProblem` entry, doctests, and the executed Literate fishing tutorial.
- Root integration, Julia 1.12.7: root `GROUP=Core` passes 61 checks (local
  controls 20, precompile workload 4, layer interface 8, multiple shooting 29);
  root `GROUP=CorleoneBase` routing passes the 334 sequential + 20 parallel
  sublibrary Core checks.
- AD, Julia 1.12.7: fails only at the ReverseDiff backend (264 pass / 4 error
  in the in-place gradient cases); ForwardDiff, Zygote, Mooncake,
  MooncakeForward, and FiniteDiff each pass 344. See Finding 1.

## Finding 1 — ReverseDiff in-place AD regression (functional)

The canonical AD group aborts at the `AutoReverseDiff` backend:

```
TrackedArrays do not support setindex!
  in dynamics at test/ad/fixtures.jl:17  (du .= -p[1] .* u)
```

Because that abort hides the remaining backends, each backend was also run
independently against the checked-out HEAD source. Only ReverseDiff fails, and
only in its in-place gradient cases:

| Backend | HEAD result |
| --- | --- |
| ForwardDiff | pass, 344 |
| ReverseDiff | fail, 264 pass / 4 error (linear/in-place and nonlinear/in-place `unprepared` + `prepared`) |
| Zygote | pass, 344 |
| Mooncake | pass, 344 |
| MooncakeForward | pass, 344 |
| FiniteDiff | pass, 344 |

Focused baseline comparison for the failing ReverseDiff backend
(`test_sequential_ad(AutoReverseDiff();
solve_kwargs = (; sensealg = SciMLSensitivity.SensitivityADPassThrough()))`
against checked-out sources, same resolved versions):

| Source | Result |
| --- | --- |
| upstream `b261940` | pass, 344 checks |
| pristine HEAD `367f073` | fail (linear/in-place and nonlinear/in-place `unprepared` + `prepared` testsets) |
| HEAD + this change | fail (same) |

Both in-place cases pass in `primal` but error in the gradient tests. This is a
genuine regression introduced by the unpushed commits, isolated from this
change's docstring/test edits. The only AD-relevant behavior change in the range
is `prepare_stage_problem`: the upstream ODE method returned in-place problems
unchanged, while the new generic `_prepare_problem` remakes every problem and
applies `aos_to_soa` to every field. It is not fixed here.

## Finding 2 — `prepare_stage_problem` always remakes (repeatable allocation)

`@benchmark CorleoneBase.prepare_stage_problem(template)` on an out-of-place
`ODEProblem` whose fields need no conversion, three runs each, matched envs:

| Source | min (ns) | median (ns) | memory (B) | allocs |
| --- | --- | --- | --- | --- |
| HEAD | 5.6–6.1 | 7.0–7.1 | 48 | 1 |
| baseline `b261940` | 1.779 | 1.78 | 0 | 0 |

Baseline early-returned the input when `u0` and `p` were unchanged; the new
implementation always `remake`s. This is a repeatable ~3.3x latency and 1
allocation (48 B) regression on a per-stage path in `step!`. It is distinct from
noise (baseline is exactly 0 allocations every run) and is not fixed here. Net
`solve` allocations did not regress because the type-stability change saves more
allocations than this costs (see Finding 4).

## Finding 3 — Parallel API is unsupported on Julia 1.10/1.11

`abstractparallel.jl` binds the ensemble hook with
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

`Base.return_types` on function barriers plus `Test.@inferred`, matched envs:

| Path | HEAD | baseline `b261940` |
| --- | --- | --- |
| `init(seq3, Tsit5; kwargs)` | concrete, `@inferred` OK | non-concrete `where {…}`, `@inferred` FAIL |
| `solve(seq1, Tsit5; kwargs)` | concrete, `@inferred` OK | non-concrete `SolutionWrapper`, `@inferred` FAIL |
| `solve(seq3, Tsit5; kwargs)` | concrete, `@inferred` OK | non-concrete `SolutionWrapper`, `@inferred` FAIL |
| `prepare_stage_problem(ode)` | concrete, `@inferred` OK | concrete, `@inferred` OK |
| `split_problem_kwargs(ode, nt)` (new) | concrete, `@inferred` OK | n/a |
| `_prepare_problem(ode)` (new) | concrete, `@inferred` OK | n/a |
| parallel `solve(par, Tsit5, EnsembleSerial; trajectories)` (new) | concrete, `@inferred` OK | n/a |

Warmed runtime (allocation) comparison over three runs, `min` (median) µs and
allocation count — no regression on the solve/init paths:

| Path | HEAD | baseline |
| --- | --- | --- |
| `solve(seq1, …)` | 56.7–57.1 (64.0–65.1) µs, 9809 allocs | 59.6–61.8 (67.6–70.6) µs, 9837 allocs |
| `solve(seq3, …)` | 122.5–123.1 (129.3–132.9) µs, 19167 allocs | 124.2–130.2 (131.5–139.3) µs, 19188 allocs |
| `init(seq3, …)` | 59.5–61.9 (64.9–65.6) µs, 9810 allocs | 62.3–66.2 (67.7–70.8) µs, 9831 allocs |

The solve/init differences are small, consistently in HEAD's favor, and
consistent with the type-stability commit; they are reported as no regression
rather than as a speedup claim. Finding 2 is the only repeatable runtime/allocation
regression found.

## New APIs without a comparable baseline

`ParallelProblem` and its `solve`, `AbstractParallelProblem`, `prob_func`,
`get_problems`, `split_problem_kwargs`, `_split_problem_kwargs`, and
`_prepare_problem` do not exist at `b261940`, so no head-vs-baseline runtime or
allocation comparison applies. Their inference is assessed above; only the
`parallel_solve` path is measured for correctness, not compared for performance.

## Coverage limitations

- The AD group cannot run green because of the ReverseDiff regression; each
  backend was therefore run independently. ForwardDiff, Zygote, Mooncake,
  MooncakeForward, and FiniteDiff pass at HEAD (344 checks each); only
  ReverseDiff fails (264 pass / 4 error), and its `primal` cases pass.
- Parallel behavior is verified only on Julia >= 1.12; on < 1.12 only the
  documented `MethodError` is asserted.
- Performance is measured on a shared workstation with BenchmarkTools' warmup
  and sampling; contention was minimised but not eliminated, so min-of-samples
  is used and only repeatable, code-explained differences are called findings.
- Inference is assessed on representative concrete call sites; this is not a
  claim that every possible usage is type-stable.

## Reproducible commands

Core (1.12 / 1.10), QA, Docs via `CORLEONE_TEST_GROUP`:

```sh
CORLEONE_TEST_GROUP=Core JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no -e 'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"])'
```

Focused AD head-vs-baseline comparison: recreate the comparison checkouts with

```sh
git worktree add --detach /tmp/opencode/cb-baseline b2619406a3cf45f871cfb9945e2f882beb4a106b
git worktree add --detach /tmp/opencode/cb-head-pristine 367f073f4493fa4f93ff65f61393655fb8de90a8
```

then `Pkg.develop` each `lib/CorleoneBase` checkout into an AD environment and run
`test/ad/runtests.jl`'s ReverseDiff call.

Inference and performance scripts and raw logs: `/tmp/opencode/inference.jl`,
`/tmp/opencode/inference_{head,base}.txt`, `/tmp/opencode/bench.jl`,
`/tmp/opencode/bench_{head,base}.txt`.
