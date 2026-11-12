# Building the root package manual

Run from the repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop([PackageSpec(path="."), PackageSpec(path="lib/CorleoneOED"), PackageSpec(path="lib/CorleoneInverse")]); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Serve the generated site locally so directory-style links resolve correctly:

```sh
julia --project=docs -e 'using LiveServer; serve(dir="docs/build", host="127.0.0.1", port=8000)'
```

Open <http://127.0.0.1:8000/> to inspect the result. LiveServer reloads the preview
when generated files change. For source watching, run this from the repository
root. Watch the authoritative sources and skip generated copies to avoid rebuild loops:

```sh
julia --project=docs -e '
using LiveServer
servedocs(
    skip_dirs = ["docs/src/examples", "docs/src/oed", "docs/src/inverse"],
    include_dirs = ["examples", "lib/CorleoneOED/examples",
        "lib/CorleoneOED/docs/src", "lib/CorleoneInverse/docs/src"],
)
'
```

Each source rebuild runs `docs/make.jl`, including its executable tutorials.

The build executes the LQR,
numerical Lotka–Volterra, ModelingToolkit Lotka–Volterra, and LTI coefficient/control
optimization tutorials, and the Lotka–Volterra universal differential equation tutorial,
plus CorleoneOED's continuous Lotka–Volterra, multiple-experiment, and discrete
compartmental tutorials, the two-stage reactor tutorial. It checks exported 
API coverage and resolves internal cross-references. External URL checks can be 
enabled with `DOCS_LINKCHECK=true julia --project=docs docs/make.jl`.

On GitHub pull requests, the build uses Documenter's `draft=true` mode unless
the PR changes `examples/`, `lib/CorleoneOED/examples/`, or
`docs/process_tutorials.jl`. This compares the PR merge commit with its first
parent, fetching that parent for shallow checkouts.
Draft builds generate tutorial Markdown without executing Literate code; outputs
and plots are omitted. Builds on `main`, tags, and local builds always execute
the full documentation. If change detection fails, a full build runs.

External documentation links use DocumenterInterLinks. Configure inventory roots
in `docs/interlinks.jl` and reference their entries with `@extref`, for example
``[`OptimizationLBFGSB.LBFGSB`](@extref Optimization)``. Optimization's inventory
also contains its solver packages, so they share the `Optimization` inventory.

MaterialDocs renders the manual with its Ocean Depth theme and a light/dark toggle.
MaterialDocs provides code highlighting throughout the manual.
`material_citations.jl` adapts the citation and bibliography nodes to MaterialDocs
so the existing references and citation anchors are preserved.
`material_images.jl` resolves Documenter's source-root figure paths relative to
each MaterialDocs output page, keeping tutorial plots visible in local and deployed builds.
DocumenterLandingPage is enabled for pages with `layout: home` YAML frontmatter
inside a `@raw html` block; its stylesheet is added automatically.
Navigation separates Getting Started, complete tutorials, a task-based Manual,
sublibraries, and the docstring API reference. CorleoneOED and its guides appear
directly under Sublibraries. Scheduling and sparse derivative
preparation have separate guides. Historical validation and deprecated constructors
retain their URLs but are hidden from the main navigation and linked from relevant pages.

The home page uses this layout for the package introduction and guide links;
the installation instructions and executable quickstart live in `getting_started.md`.

Dependency setup is separate from the build: `make.jl` uses the active environment
without developing or resolving packages again. Set `CI=false` for a local build
if your shell otherwise sets `CI=true`; deployment runs only in CI.

Edit tutorials in their `examples/<name>/main.jl` sources, not the generated
Markdown. DemoCards stages the authoritative scripts and executes each tutorial
once through Literate; Documenter includes its rendered output. Introductory
guide examples and doctests execute through Documenter.

The manual documents the checked-out Corleone, CorleoneOED, and CorleoneInverse
modules. OED tutorials
are executed from `lib/CorleoneOED/examples/*/main.jl`; its guide and API pages are
generated from `lib/CorleoneOED/docs/src/`. Edit those authoritative sources, not
`docs/src/oed/`. Both sites share the tutorial sources without handwritten Markdown
duplicates. The root site retains main's established OED tutorial URLs.
CorleoneInverse's overview and API pages are generated from
`lib/CorleoneInverse/docs/src/`; edit those sources rather than `docs/src/inverse/`.

The generated manifest, tutorial Markdown and figures, and HTML output are
ignored by Git. If a manifest from an older checkout references a removed local
package, remove `docs/Manifest.toml` and run the setup command again.

DemoCards.jl builds the root-site tutorials via `docs/process_tutorials.jl`.
Its temporary staging directory is ignored; edit the original example scripts.
The Tutorials overview uses DemoCards’ `bokehlist` theme with Optimal Control,
Optimal Experimental Design, and Parameter Estimation sections. The generated
DemoCards stylesheet is loaded by MaterialDocs.

Card metadata uses native DemoCards YAML frontmatter, with a leading `# ` on each
line between `# ---` delimiters. Set `title`, `description`, `id` (the generated
page’s heading anchor), and `julia` (the minimum Julia version). Keep
`notebook: false` and `generate_cover: false` so each tutorial executes once.
The build sets `execute` according to its draft mode.

The optional project-specific `category` selects a gallery section; root examples
otherwise use Optimal Control and CorleoneOED examples use Optimal Experimental
Design. The optional `cover_figure` selects a numbered figure from the tutorial’s
executed output (default: 1). The gallery copies this plot as its cover without
executing the script again. Draft builds use the DemoCards placeholder when
executed figures are unavailable. The standalone CorleoneOED manual strips the
card frontmatter and generates the same heading before processing its tutorials
with Literate.
