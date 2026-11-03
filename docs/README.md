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

On GitHub pull requests, the build uses Documenter's `draft=true` mode unless
the PR changes a tutorial under `examples/` or `lib/CorleoneOED/examples/`, the
benchmark tutorial page, or `docs/process_tutorials.jl`. This compares the PR
merge commit with its first parent, fetching that parent for shallow checkouts.
Draft builds generate tutorial Markdown without executing Literate code; outputs
and plots are omitted. Builds on `main`, tags, and local builds always execute
the full documentation. If change detection fails, a full build runs.

External documentation links use DocumenterInterLinks. Configure inventory roots
in `docs/interlinks.jl` and reference their entries with `@extref`, for example
``[`OptimizationLBFGSB.LBFGSB`](@extref Optimization)``. Optimization's inventory
also contains its solver packages, so they share the `Optimization` inventory.

MaterialDocs renders the manual with its Ocean Depth theme and a light/dark toggle.
DocumenterCodeBlocks remains registered, but currently only enhances
`Documenter.HTML` output; MaterialDocs provides its own code highlighting.
`material_citations.jl` adapts the citation and bibliography nodes to MaterialDocs
so the existing references and citation anchors are preserved.
DocumenterLandingPage is enabled for pages with `layout: home` YAML frontmatter
inside a `@raw html` block; its stylesheet is added automatically.
The home page uses this layout for the package introduction and guide links;
the installation instructions and executable quickstart live in `getting_started.md`.

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
