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

GitHub Actions persists `docs/pluto_output/`, the output directory used by
DemoCards and PlutoStaticHTML. Its cache key includes the resolved documentation
manifest, notebook sources, and local package sources. Changed dependencies or
package code therefore trigger fresh notebook evaluation in CI.

Local builds use PlutoStaticHTML's normal cache. Remove `docs/pluto_output/` when
you want fresh results after local package edits. No custom cache code is used.
The resolved manifest, rendered notebook Markdown, and built site are ignored by
Git. Set `DOCS_LINKCHECK=true` to enable external URL checks.

Serve the built site to inspect directory-style links:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs/build
```

Open <http://127.0.0.1:8000/>. Local builds do not deploy; `deploydocs` runs only in CI.
