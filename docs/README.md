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

A full build executes uncached tutorials and every `@example` block, runs the
doctests, checks that all exported names are documented, and resolves
cross-references. Unchanged tutorials reuse their previously executed output.
Useful environment variables:

| Variable | Effect |
|---|---|
| `DOCS_DRAFT=true` | Draft build: skips all code execution, so tutorials have no output or plots. Much faster for checking prose and links. |
| `DOCS_TUTORIAL_CACHE=false` | Execute every tutorial afresh without reading or writing the tutorial cache. |
| `DOCS_LINKCHECK=true` | Also checks external URLs. |
| `CI=false` | Never deploy. Deployment only runs when `CI=true`. |

On GitHub pull requests, the build is a draft unless the PR changes `examples/`,
`lib/CorleoneOED/examples/`, `lib/CorleoneInverse/examples/`, or
`docs/process_tutorials.jl`, `docs/tutorial_cache.jl`, or `docs/tutorials.toml`.
Full builds on pull requests, `main`, tags, and local runs use the per-tutorial
cache unless `DOCS_DRAFT` or `DOCS_TUTORIAL_CACHE` overrides it.

## Tutorial execution cache

Each tutorial has an independent entry in `docs/.cache/tutorials/`, containing its
executed Markdown, downloadable Julia script, and figure assets. The GitHub Actions
workflow restores this cache separately from Julia's package cache. The archive
transports the entries together, but each tutorial validates its own fingerprint;
editing one example does not invalidate the rest of the folder.

Fingerprints include the example directory (including local data), declared helper
paths, all checked-out package source and extension code, package projects, the docs
project and resolved manifest, the rendering scripts, Julia version, OS,
architecture, and thread count. Changing package code or the resolved environment
invalidates all tutorial entries. Editing a handwritten manual page does not.
Checkout relocation is ignored for the three checked-out package paths in the
manifest. Cached dependency/system appendices describe the original execution.

Incomplete or modified cache payloads are treated as misses. Only successfully
executed tutorials are stored; draft builds neither read nor overwrite executed
entries. Pages and the gallery are always regenerated, so removed tutorials cannot
leave stale pages behind. The cache keeps one successful result per tutorial.

To force a fresh numerical run, use:

```sh
DOCS_TUTORIAL_CACHE=false julia --project=docs docs/make.jl
```

Files inside an example's own directory are tracked automatically. Declare helpers
or data outside it in the tutorial's commented YAML frontmatter, relative to that
example's directory:

```julia
# cache_dependencies:
#   - ../shared/model.jl
#   - ../shared/measurements.tsv
```

For example, the multiple-shooting tutorial declares the design helper from the
OED-control tutorial. Add declarations when introducing cross-directory inputs.
Changes to external services or undeclared inputs require a forced fresh run.
Cache hits and misses are logged by tutorial name.

The cache integration tests run two tiny tutorials through the real DemoCards
pipeline and verify independent invalidation, environment/code changes, corrupted
assets, gallery covers, draft isolation, and forced execution:

```sh
julia --project=docs docs/test/runtests.jl
```

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
2. Add the repository-relative script path and its output name to `docs/tutorials.toml`.
   The output name fixes the page URL, so don't change it once published.
3. Add the page to the navigation in `docs/make.jl`.

On a cache miss, each tutorial runs once through Literate, and its plots are saved
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
