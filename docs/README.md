# Building the root package manual

Run from the repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop([PackageSpec(path="."), PackageSpec(path="lib/CorleoneOED"), PackageSpec(path="lib/OptimalControlBenchmarks")]); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Serve the generated site locally so directory-style links resolve correctly:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/build
```

Open <http://127.0.0.1:8000/> to inspect the result. The build executes the LQR,
numerical Lotka–Volterra, ModelingToolkit Lotka–Volterra, and LTI coefficient/control
optimization tutorials, plus CorleoneOED's continuous Lotka–Volterra, multiple-experiment,
and discrete compartmental tutorials. It checks exported API coverage and resolves
internal cross-references. External URL checks can be enabled with
`DOCS_LINKCHECK=true julia --project=docs docs/make.jl`.

Dependency setup is separate from the build: `make.jl` uses the active environment
without developing or resolving packages again. Set `CI=false` for a local build
if your shell otherwise sets `CI=true`; deployment runs only in CI.

Edit tutorials in their `examples/<name>/main.jl` sources, not the generated
Markdown. Literate executes each tutorial once and Documenter includes its rendered
output. Introductory guide examples and doctests execute through Documenter.

The manual documents the checked-out Corleone, CorleoneOED, and
OptimalControlBenchmarks modules. OED tutorials
are executed from `lib/CorleoneOED/examples/*/main.jl`; its guide and API pages are
generated from `lib/CorleoneOED/docs/src/`. Edit those authoritative sources, not
`docs/src/oed/`. Both sites share the tutorial sources without handwritten Markdown
duplicates. The root site retains main's established OED tutorial URLs.

The Optimal Control Benchmarks section is generated from
`lib/OptimalControlBenchmarks/docs/src/index.md` and documents the checked-out
OptimalControlBenchmarks module. Edit that source rather than
`docs/src/benchmarks/`. Its LQR example executes during the documentation build.

The generated manifest, tutorial Markdown and figures, and HTML output are
ignored by Git. If a manifest from an older checkout references a removed local
package, remove `docs/Manifest.toml` and run the setup command again.
