# Building the root package manual

Run from the repository root:

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
```

Open `docs/build/index.html` to inspect the result. The build executes both root
`ShootingLayer` tutorials, checks exported API coverage, and resolves internal
cross-references. External URL checks can be enabled with
`DOCS_LINKCHECK=true julia --project=docs docs/make.jl`.

The manual documents the checked-out root `src` module. It does not load packages
under `lib` or execute their tutorials. The legacy ModelingToolkit example is
also excluded until it is migrated to the current root API.

The generated manifest, tutorial Markdown and figures, and HTML output are
ignored by Git. If a manifest from an older checkout references a removed local
package, remove `docs/Manifest.toml` and run the setup command again.
