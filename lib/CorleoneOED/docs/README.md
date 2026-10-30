# Building the CorleoneOED manual

From `lib/CorleoneOED`, develop both checked-out packages and install the documentation dependencies:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop([PackageSpec(path="../.."), PackageSpec(path=".")]); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Explicit development also supports Julia 1.10, which does not use `[sources]` entries.
The build checks that both packages come from this checkout. Dependency resolution
is separate from `make.jl`; the build does not install packages or publish a site.
Keep the resolved `docs/Manifest.toml`, `julia --version`, and `Pkg.status()` output
with a build log when reproducing exact numerical results. Compat bounds do not
freeze every dependency version; the tutorials validate properties rather than
requiring identical printed floating-point values across environments.

The documentation environment includes the runtime packages and ODE solver, plus
Literate and Documenter. ComponentArrays provides the tutorials' vectorizer;
Optimization, OptimizationMOI, and Ipopt provide their continuous solver frontend;
CairoMakie renders figures without a display server. These are documentation/example
dependencies, not additional CorleoneOED runtime dependencies.

Every build executes all three `examples/*/main.jl` tutorials through
`Literate.markdown(execute=true)`, then runs Documenter examples, doctests,
cross-reference checks, and exported API coverage. Tutorial assertions check solver
termination, finite positive-definite information, and budget feasibility. There
is no option to substitute stale generated tutorial output for execution.

Edit those Julia tutorial sources, not `docs/src/tutorials/*.md`. Generated Markdown,
PNG figures, HTML, and the local manifest are ignored by Git. A build failure stops
execution; do not suppress an assertion or relax a reference to publish a page.

Serve the site for visual inspection:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/build
```

Open <http://127.0.0.1:8000/>. Inspect the mathematical formulas, field documentation,
tutorial plots and captions, navigation, and API signatures. Optional external
URL checking uses `DOCS_LINKCHECK=true julia --project=docs docs/make.jl`.

After the setup above, run the configured OED groups from `lib/CorleoneOED`:

```sh
GROUP=Core julia --project=docs -e 'using Pkg; Pkg.test("CorleoneOED")'
GROUP=QA julia --project=docs -e 'using Pkg; Pkg.test("CorleoneOED")'
```

`Pkg.test` creates an isolated test environment and uses the package's group
runner. Keep both packages developed in the invoking environment so tests use the
checked-out root contracts. Root tests need rerunning when shared numerical or
optimization contracts change; editorial changes alone do not require another
full root run.
