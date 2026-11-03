# Building the CorleoneParameterEstimation manual

From `lib/CorleoneParameterEstimation`, develop all checked-out packages and install the
documentation dependencies:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop([PackageSpec(path="../.."), PackageSpec(path="../../lib/CorleoneOED"), PackageSpec(path=".")]); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Explicit development also supports Julia 1.10, which does not use `[sources]` entries.
The build checks that the packages come from this checkout. Dependency resolution is
separate from `make.jl`; the build does not install packages or publish a site. Keep the
resolved `docs/Manifest.toml`, `julia --version`, and `Pkg.status()` output with a build
log when reproducing exact numerical results. Compat bounds do not freeze every dependency
version; the tutorials validate properties rather than requiring identical printed
floating-point values across environments.

The documentation environment includes the runtime packages and ODE solvers, plus Literate
and Documenter. CorleoneOED is needed only by the fixed-control tutorial. ComponentArrays
provides the tutorials' vectorizer; Optimization, OptimizationMOI, OptimizationOptimisers,
OptimizationLBFGSB, and Ipopt provide their solver frontends; Lux and LuxCore provide the
universal differential equation; CairoMakie renders figures without a display server. These
are documentation/example dependencies, not additional CorleoneParameterEstimation runtime
dependencies.

Every build executes all wired `examples/*/main.jl` tutorials through
`Literate.markdown(execute=true)`, then runs Documenter examples, doctests,
cross-reference checks, and exported API coverage. There is no option to substitute stale
generated tutorial output for execution.

Edit those Julia tutorial sources, not `docs/src/tutorials/*.md`. Generated Markdown, PNG
figures, HTML, and the local manifest are ignored by Git. A build failure stops execution;
do not suppress an assertion or relax a reference to publish a page.

Serve the site for visual inspection:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/build
```

Open <http://127.0.0.1:8000/>. Optional external URL checking uses
`DOCS_LINKCHECK=true julia --project=docs docs/make.jl`.

After the setup above, run the configured groups from `lib/CorleoneParameterEstimation`:

```sh
CORLEONE_TEST_GROUP=Core julia --project=docs -e 'using Pkg; Pkg.test("CorleoneParameterEstimation")'
CORLEONE_TEST_GROUP=QA julia --project=docs -e 'using Pkg; Pkg.test("CorleoneParameterEstimation")'
```

`Pkg.test` creates an isolated test environment and uses the package's group runner. Keep
the packages developed in the invoking environment so tests use the checked-out root
contracts.
