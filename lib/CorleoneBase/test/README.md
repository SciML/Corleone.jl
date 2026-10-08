# CorleoneBase test environments

`All` (the default) runs only Core. `Core`, `AD`, and `QA` are independently
selectable through `CORLEONE_TEST_GROUP`, falling back to `GROUP`. Bare
`CorleoneBase` selects Core; `CorleoneBase_AD` and `CorleoneBase_QA` select their
named groups. `test_groups.toml` supplies all three centralized sublibrary CI
lanes. No workflow changes are needed.

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
AD execution is still in progress; no complete AD result is claimed yet.

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
