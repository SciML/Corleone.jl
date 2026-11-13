# Building the documentation

The manual documents the checked-out Corleone, CorleoneOED, and CorleoneInverse
packages. All commands run from the repository root.

## Setup

Develop the local packages into the documentation environment once:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop([PackageSpec(path="."), PackageSpec(path="lib/CorleoneOED"), PackageSpec(path="lib/CorleoneInverse")]); Pkg.instantiate()'
```

`make.jl` uses this environment as is and never resolves packages itself. If a
manifest from an older checkout fails to load, delete `docs/Manifest.toml` and run
the setup again.

## Building

```sh
julia --project=docs docs/make.jl
```

A full build executes every tutorial and `@example` block, runs the doctests,
checks that all exported names are documented, and resolves cross-references.
Useful environment variables:

| Variable | Effect |
|---|---|
| `DOCS_DRAFT=true` | Draft build: skips all code execution, so tutorials have no output or plots. Much faster for checking prose and links. |
| `DOCS_LINKCHECK=true` | Also checks external URLs. |
| `CI=false` | Never deploy. Deployment only runs when `CI=true`. |

On GitHub pull requests, the build is a draft unless the PR changes `examples/`,
`lib/CorleoneOED/examples/`, `lib/CorleoneInverse/examples/`, or
`docs/process_tutorials.jl`. Builds on `main`, tags, and local builds execute
everything unless `DOCS_DRAFT` is set.

## Previewing

Serve the generated site so that directory-style links resolve:

```sh
julia --project=docs -e 'using LiveServer; serve(dir="docs/build", host="127.0.0.1", port=8000)'
```

and open <http://127.0.0.1:8000/>. To rebuild on changes, watch the authoritative
sources and skip the generated copies:

```sh
julia --project=docs -e '
using LiveServer
servedocs(
    skip_dirs = ["docs/src/examples", "docs/src/oed", "docs/src/inverse"],
    include_dirs = ["examples", "lib/CorleoneOED/examples", "lib/CorleoneInverse/examples",
        "lib/CorleoneOED/docs/src", "lib/CorleoneInverse/docs/src"],
)
'
```

## Where the sources live

Several directories under `docs/src` are generated during the build and ignored by
Git. Edit the sources instead:

| Generated | Source |
|---|---|
| `docs/src/examples/`, `docs/src/gallery/` | `examples/*/main.jl`, `lib/CorleoneInverse/examples/*/main.jl`, `lib/CorleoneOED/examples/*/main.jl` |
| `docs/src/oed/` | `lib/CorleoneOED/docs/src/` (also used by CorleoneOED's own manual) |
| `docs/src/inverse/` | `lib/CorleoneInverse/docs/src/` |

All other pages in `docs/src` are handwritten. When copying the OED pages, the build
rewrites their GitHub links to Corleone guides into local links.

## Writing tutorials

Tutorials are [Literate](https://github.com/fredrikekre/Literate.jl) scripts that
[DemoCards](https://github.com/JuliaDocs/DemoCards.jl) arranges into the gallery.
To add one:

1. Create `main.jl` in the example folder, starting with DemoCards frontmatter:

   ```julia
   # ---
   # title: Short title in sentence case
   # description: One sentence for the gallery card
   # id: unique_anchor
   # julia: "1.10"
   # notebook: false
   # generate_cover: false
   # ---
   ```

   `id` is the anchor for `@ref` links. Keep `notebook: false` and
   `generate_cover: false` so each tutorial executes only once. Optional keys:
   `category` (`optimal_control`, `parameter_estimation`, or
   `experimental_design`; the default depends on the folder) and `cover_figure`
   (which plot to use as the gallery cover, default 1).
2. Add the script and its output name to the list in `docs/process_tutorials.jl`.
   The output name fixes the page URL, so don't change it once published.
3. Add the page to the navigation in `docs/make.jl`.

Each tutorial runs once through Literate during the build, and its plots are saved
as PNG files next to the page. The build appends a dependency and system summary
to every tutorial.

## Other configuration

- External `@extref` links use DocumenterInterLinks; inventories are configured in
  `docs/interlinks.jl`. Optimization's inventory also covers its solver packages,
  for example ``[`OptimizationLBFGSB.LBFGSB`](@extref Optimization)``.
- The home page uses DocumenterLandingPage: its hero and feature cards are YAML
  frontmatter in a `@raw html` block in `index.md`.
- The site uses Documenter's HTML theme. `material_citations.jl` and
  `material_images.jl` adapt citations and figure paths for MaterialDocs, which is
  currently disabled in `make.jl`.
