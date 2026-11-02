# Building the manual and tutorial galleries

Run from the repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop([PackageSpec(path="."), PackageSpec(path="lib/CorleoneOED"), PackageSpec(path="lib/OptimalControlBenchmarks")]); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

The galleries use DemoCards' `PlutoDemoCard` support and PlutoStaticHTML's static
Documenter output. The released DemoCards extension supports PlutoStaticHTML 6
and Pluto 0.19; CI uses Julia 1.10 with these compatible versions.

Edit the authoritative Pluto notebooks in `docs/examples/` and
`lib/CorleoneOED/docs/oed_examples/`. To open the control tutorials interactively:

```sh
julia --project=docs -e 'using Pluto; Pluto.run()'
```

Open a notebook from `docs/examples/` in Pluto. Each notebook activates the nearest
project above its source directory, following Pluto's shared-environment pattern.
The notebooks also run as ordinary Julia scripts. For the OED notebooks, prepare
[their documentation environment](../lib/CorleoneOED/docs/README.md) first.
The existing `examples/*/main.jl` entrypoints include these notebook sources;
`examples/install_deps.jl` prepares both documentation environments.

DemoCards renders the notebooks, includes their static results, and provides
source downloads. Documenter builds the guide/API pages, doctests, and references.
The root manual includes the OED guide from `lib/CorleoneOED/docs/src/` and the
benchmark guide from `lib/OptimalControlBenchmarks/docs/src/index.md`; edit those
sources rather than generated copies under `docs/src/`.

## Cached results

The reusable SciML documentation workflow remains responsible for building and
deploying the site. Following [PlutoStaticHTML's `previous_dir` pattern](https://juliapluto.github.io/PlutoStaticHTML.jl/stable/#Caching), each build
copies `docs/pluto_output/` into the deployed site. CI restores these Markdown
outputs from `dev/pluto_output/` on `gh-pages` before calling `makedemos`.
Missing caches fall back to notebook evaluation. The workflow also enables its
standard Julia package cache.

PlutoStaticHTML validates the notebook checksum and Julia version. These notebooks
use a shared project, so edits to package code or dependencies also require fresh
outputs: remove `docs/pluto_output/` locally or the corresponding deployed cache
files in `dev/pluto_output/` on `gh-pages` before rebuilding.
The resolved manifest, rendered notebook Markdown, and built site are ignored by
Git. Set `DOCS_LINKCHECK=true` to enable external URL checks.

Serve the built site to inspect directory-style links:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/build
```

Open <http://127.0.0.1:8000/>. Local builds do not deploy; `deploydocs` runs only in CI.
