# CorleoneBase test environments

`All` (the default) runs only Core. `Core`, `AD`, and `QA` are independently
selectable through `CORLEONE_TEST_GROUP`, falling back to `GROUP`. Bare
`CorleoneBase` selects Core; `CorleoneBase_AD` and `CorleoneBase_QA` select their
named groups. `test_groups.toml` supplies all three centralized sublibrary CI
lanes. No workflow changes are needed.

`Docs` is an additional opt-in isolated lane (`CorleoneBase_Docs` through the
root dispatcher). It strictly renders the CorleoneBase pages from the shared
manual, runs their doctests against the checked-out source, executes the Literate
fishing tutorial, and independently reproduces the optimization with stage,
bound, continuity, objective-improvement, and directional-gradient checks.
It does not add documentation, optimization, plotting, or solver packages to
runtime dependencies, or change the default Core-only `All` selection.

Core runs two lanes: `Sequential Core contracts` (`core/sequential.jl`) and
`Parallel Core contracts` (`core/parallel.jl`). The parallel lane covers the
`ParallelProblem`/`AbstractParallelProblem` API added by the unpushed commits:
construction, the default and custom `prob_func` context contract,
ensemble-versus-solve keyword routing, `output_func`, reproducibility through
`seed`, and the required `trajectories` keyword. The ensemble hook is bound with
`Base.Fix1(prob_func, problem)` and SciMLBase calls it with two arguments;
`Base.Fix1` only forwards multiple remaining arguments on Julia >= 1.12, so the
parallel entry point cannot dispatch on the declared 1.10/1.11 minimum. The lane
asserts the resulting `MethodError` on < 1.12 instead of skipping the API, so
Core stays green on 1.10 while the limitation remains visible.

The sequential lane adds regression coverage for the same commits: the abstract
problem now subtypes `SciMLBase.AbstractSciMLProblem`, `init` dispatches only on
`SciMLBase.AbstractSciMLAlgorithm`, and `init`/`solve` keywords are partitioned
by `fieldnames(typeof(problem))` instead of the previous hard-coded
`u0`/`p`/`tspan` set. `prepare_stage_problem` now covers every
`SciMLBase.AbstractSciMLProblem` field rather than only `ODEProblem` `u0`/`p`,
returns in-place problems unchanged, and returns the input itself when no field
needs conversion. See `REGRESSION_FINDINGS.md` for the type-inference and
performance comparison against the upstream baseline `b261940`.

```sh
CORLEONE_TEST_GROUP=Docs JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no -e 'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"])'
```

For a browsable persistent focused build, run the setup command in
`docs/check_corleonebase.jl` with `CORLEONEBASE_DOCS_OUTPUT` set to an absolute
output directory. The full shared manual uses `docs/make.jl`. See
`docs/src/corleonebase.md` for standalone tutorial setup and numerical criteria.

The documentation change was regression-tested on Julia 1.12.7: Core passed
302 checks, AD 2,143, QA 23, and root Core 61. A fresh base-only environment
also verified that documentation, optimization, plotting, solver, AD-backend,
and QA packages were absent, and the optional Zygote extension was unloaded.
The standalone tutorial passed on Julia 1.10.12 against checked-out sources:
24 successful stages, starting cost 7.21591176555, optimized cost 1.34750999922,
and control extrema (1.36484e-7, 0.999998273). Optimizer retcode was `Success`
with requested tolerance 1e-6; ODE absolute and relative tolerances were 1e-9.
Reevaluation at 1e-11 gave cost 1.34750999927. These are observed numerical
results for a local relaxed 24-control optimization, not a global certificate.

The initial Docs attempt exposed a Literate prose comment inside a function;
it is now a `##` code comment so the function remains one executable chunk.
Subsequent doctests exposed the need to explicitly import `successful_retcode`
and the solver's rejection of Boolean `verbose`. The failure example now uses
the standard logger to silence its intentional warning, retaining the exact
return-code and no-transition checks. No assertions or strict doctest checks
were removed to accommodate these failures.

The final focused HTML build and checked-source doctests pass without warnings.
The Docs lane passes 112 checks through root `GROUP=CorleoneBase_Docs` routing,
including independent source execution, repeated objectives, all stage endpoints
and controls, continuity, and a ForwardDiff directional-gradient comparison to
central differences (atol 1e-5, rtol 1e-4). On Julia 1.12.7 the independent run
gave starting cost 7.21591176555, optimized cost 1.34750999944, tightened cost
1.34750999949, and successful ODE and optimizer return codes. HTTP checks also
verified 68 local navigation/anchor links across the rendered guide, API, and
tutorial pages. The focused build includes a working landing page and uses
Documenter-flavored Literate output so the tutorial's reference anchor survives;
the existing tutorials retain their original rendering flavor.
The same strict page render, doctests, and executed Literate tutorial also pass
on Julia 1.10.12 in a fresh environment developed against the checked-out source.

The full shared `docs/make.jl` build also passes with the latest guide, API, and
tutorial, all six existing tutorials executed, doctests enabled, exported API
coverage checked, and link checking enabled. HTTP verification of the full
manual checks five pages and 171 navigation/anchor links, including home/API
discovery, the latest stage-termination doctest, and the numerical report.
The termination doctest passes on Julia 1.10.12 and 1.12.7: a successful
`ReturnCode.Terminated` stage does not itself stop the sequence. Existing
bibliography/navbar/deployment warnings are non-fatal and left unrepaired;
local deployment is correctly skipped. Literate edit metadata is pinned to the
locally confirmed `origin/HEAD` (`main`) to avoid network branch discovery.

## Dependency placement

- Runtime: ArrayInterface converts AD array representations with `aos_to_soa`;
  CommonSolve supplies the solve/init/step! interfaces; SciMLBase supplies problem
  types, remake, and return-code checks.
- Removed direct dependencies: ConcreteStructs, DocStringExtensions, Random,
  Reexport, and SymbolicIndexingInterface have no source or extension uses.
  Some remain transitive dependencies of SciMLBase or the test solver.
- OrdinaryDiffEqTsit5 is a Core test solver, not part of the runtime API. It is
  also declared in the AD environment, which tests real ODE gradients.
- ChainRulesCore and Zygote remain weakdeps and joint extension triggers. The
  AD and QA environments declare them directly to exercise the extension.
- DifferentiationInterface, FiniteDiff, ForwardDiff, ReverseDiff, Mooncake, and
  SciMLSensitivity live only in `ad/Project.toml`. Existing backend and analytic
  gradient coverage is retained, including prepared/repeated evaluations.
- Aqua is declared only in `qa/Project.toml`. SafeTestsets and SciMLTesting are
  test-harness dependencies, never runtime dependencies. SciMLTesting itself
  depends on Aqua/ExplicitImports, so ordinary `Pkg.test` installs those tools
  even when QA is not selected. The Core **body** also runs without that harness
  or either tool, as verified below. The solver transitively depends on
  DifferentiationInterface/EnzymeCore, but does not require the AD backends.

Every direct dependency has compat in its owning environment; shared bounds
match the package declarations. The package version is unchanged.

## Verification commands

Run from the repository root. Scratch projects avoid the package's existing
ignored Manifest and preserve local environments. All commands disable startup
files and develop the checked-out source, rather than a registered release.

For each of `Core`, `AD`, `QA`, and `All` (replace `Core` below):

```sh
CORLEONE_TEST_GROUP=Core JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no -e 'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"])'
```

The Julia 1.10 Core compatibility check uses the same command with `julia +1.10`
instead of `julia`.

Base-only loading and absence of optional/test dependencies:

```sh
JULIA_LOAD_PATH=@:@stdlib JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no -e 'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.instantiate(); using CorleoneBase; @assert realpath(pkgdir(CorleoneBase)) == realpath("lib/CorleoneBase"); names = Set(info.name for info in values(Pkg.dependencies())); @assert isempty(intersect(names, Set(["Zygote", "ChainRulesCore", "DifferentiationInterface", "ForwardDiff", "ReverseDiff", "Mooncake", "SciMLSensitivity", "Aqua", "SciMLTesting", "OrdinaryDiffEqTsit5"]))); @assert Base.get_extension(CorleoneBase, :CorleoneBaseZygoteExtension) === nothing; println("BASE ISOLATION PASSED")'
```

Core without AD backends or QA tools (including no global environment fallback):

```sh
JULIA_LOAD_PATH=@:@stdlib JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no -e 'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=abspath("lib/CorleoneBase")); Pkg.add(["CommonSolve", "OrdinaryDiffEqTsit5", "SciMLBase", "Test"]); Pkg.instantiate(); names = Set(info.name for info in values(Pkg.dependencies())); @assert isempty(intersect(names, Set(["Zygote", "ChainRulesCore", "ForwardDiff", "ReverseDiff", "Mooncake", "SciMLSensitivity", "Aqua", "SciMLTesting", "ExplicitImports"]))); using CorleoneBase; @assert realpath(pkgdir(CorleoneBase)) == realpath("lib/CorleoneBase"); include("lib/CorleoneBase/test/core/sequential.jl"); @assert Base.get_extension(CorleoneBase, :CorleoneBaseZygoteExtension) === nothing; println("CORE WITHOUT AD/QA PASSED")'
```

Root integration tests:

```sh
GROUP=Core JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no -e 'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop(path=pwd()); Pkg.test("Corleone"; julia_args=["--startup-file=no"])'
```

Default All, explicit Core, and root-to-sublibrary routing (the exact combined
command used for these checks):

```sh
JULIA_PKG_PRECOMPILE_AUTO=0 julia --startup-file=no -e 'using Pkg; Pkg.activate(mktempdir("/tmp/opencode")); Pkg.develop([PackageSpec(path=abspath("lib/CorleoneBase")), PackageSpec(path=pwd())]); for group in ("All", "Core"); withenv("CORLEONE_TEST_GROUP"=>group) do; Pkg.test("CorleoneBase"; julia_args=["--startup-file=no"]); end; end; withenv("GROUP"=>"CorleoneBase", "CORLEONE_TEST_GROUP"=>nothing) do; Pkg.test("Corleone"; julia_args=["--startup-file=no"]); end'
```

QA asserts the extension is absent before its triggers load, still absent with
only ChainRulesCore, and present after Zygote loads. It runs SciMLTesting.run_qa
with Aqua enabled and no Aqua exceptions. ExplicitImports exceptions are limited
to the documented Zygote reexport/unmarked API, the existing keyword dispatch
hook, Base's NamedTuple key subtraction, and private parent-extension hooks;
see `qa/qa.jl`. Docstrings are checked, but rendering is not: this sublibrary has
no standalone manual. JET is not configured, matching root QA.

The first QA attempt exposed the import/API ownership exceptions above; the
second exposed the remaining unmarked Zygote/keyword APIs. No check was marked
broken or disabled to hide those failures. The first Core run exposed a test
fixture missing a return-code field: the fixture now has a real return code and
the test matches the intended initialization failure message.

The Julia 1.10.12 Core attempt found that the original multi-argument `Base.Fix1`
callbacks require Julia 1.12 (a standalone check confirmed the limitation on
1.11.7 too). Both ordinary initialization and the Zygote initialization rule
now bind the parent problem with equivalent two-argument closures. Core then
passed all 233 checks on Julia 1.10.12, without dropping any test or raising the
declared minimum Julia version. Full AD/QA compatibility on older Julia versions
and every version permitted by the dependency bounds is not established by that
Core run.

## Observed results

On Julia 1.12.7, Core and All each pass 233 checks; QA passes 23 checks (Aqua,
all six ExplicitImports checks, docstrings, reexports, checked-out package path,
and staged extension loading). Root Core passes 61 checks: local controls (20),
precompile workload (4), layer interface (8), and multiple shooting (29).
Root `GROUP=CorleoneBase` routing also passes the 233-check sublibrary Core suite.
These were rerun after the callback compatibility fix. The isolated Core body
and base-only dependency assertions pass. Julia 1.10.12 Core passes 233 checks.
The final AD run passes all 2,143 checks, with no skips or broken tests, including
ForwardDiff, ReverseDiff, Zygote, Mooncake reverse/forward, finite differences,
and supplemental Zygote rule contracts. It resolves DifferentiationInterface
0.7.21, ForwardDiff 1.4.6, ReverseDiff 1.18.4, Zygote 0.7.13, Mooncake 0.5.63,
FiniteDiff 2.33.0, ChainRulesCore 1.26.1, and SciMLSensitivity 7.119.12 against
the checked-out CorleoneBase 0.1.0. Common runtime/test versions are ArrayInterface
7.30.2, CommonSolve 0.2.14, SciMLBase 3.57.0, OrdinaryDiffEqTsit5 2.1.5,
SafeTestsets 0.1.0, SciMLTesting 2.13.2, and Aqua 0.8.18.

The final combined All/Core/routing run also appended a root Core run inside the
same driver, after the routing block:

```julia
withenv("GROUP"=>"Core", "CORLEONE_TEST_GROUP"=>nothing) do
    Pkg.test("Corleone"; julia_args=["--startup-file=no"])
end
```

The root-routing check warns that the pre-existing ignored sublibrary Manifest
is stale, but Pkg's test sandbox resolves the new declarations and passes. That
local Manifest is deliberately not rewritten or committed. Fresh scratch
resolves avoid this warning and independently verify the new dependency graph.

## Current results after the unpushed API changes

On Julia 1.12.7, Core passes 334 sequential checks and 20 parallel checks; root
`GROUP=CorleoneBase` routing passes the same 354 checks, and root `GROUP=Core`
passes 61. QA passes 23, and Docs passes 112 on both Julia 1.12.7 and 1.10.12.
On Julia 1.10.12, Core passes 334
sequential checks and 6 parallel checks (the parallel lane asserts the
documented `MethodError`). Type inference on the changed `init`/`solve` paths
is concrete and `@inferred`-passing, whereas the upstream baseline was not.
All findings, evidence, and coverage limitations are in
`REGRESSION_FINDINGS.md`.
