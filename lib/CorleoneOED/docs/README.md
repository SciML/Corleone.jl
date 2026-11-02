# Building the CorleoneOED manual

From the repository root, prepare the shared notebook environment and build:

```sh
julia --project=lib/CorleoneOED/docs -e 'using Pkg; Pkg.develop([PackageSpec(path="."), PackageSpec(path="lib/CorleoneOED")]); Pkg.instantiate()'
julia --project=lib/CorleoneOED/docs lib/CorleoneOED/docs/make.jl
```

Use Julia 1.10 with the documentation environment's DemoCards 0.5,
PlutoStaticHTML 6, and Pluto 0.19. To edit the notebooks interactively:

```sh
julia --project=lib/CorleoneOED/docs -e 'using Pluto; Pluto.run()'
```

Open a notebook from `lib/CorleoneOED/docs/oed_examples/`. These are the same
sources used by the root manual's OED gallery. They activate the nearest shared
project and can also be executed as ordinary Julia scripts.

DemoCards renders `PlutoDemoCard`s through PlutoStaticHTML. Static results are
cached in `lib/CorleoneOED/docs/pluto_output/`; remove this directory for a fresh
local run after editing package code. The root site's GitHub Actions workflow
persists its own `docs/pluto_output/` cache between matching builds.

The build checks exported API documentation, internal references, doctests, and
notebook execution. Set `DOCS_LINKCHECK=true` to check external URLs. It builds
locally without deploying. Serve `lib/CorleoneOED/docs/build/` with an HTTP server
to inspect the result.

Edit guide/API pages in `docs/src/` and notebooks in `docs/oed_examples/` within
the sublibrary. Rendered pages and manifests are generated artifacts.
