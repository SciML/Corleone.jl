# Sublibraries

## CorleoneInverse

CorleoneInverse fits shared model parameters and experiment-specific controls or
initial states to observations. From the repository root, set up the example
environment and run the introductory Lotka–Volterra fit:

```sh
julia --project=lib/CorleoneInverse/examples -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path="."), Pkg.PackageSpec(path="lib/CorleoneInverse"), Pkg.PackageSpec(path="lib/CorleoneOED")]); Pkg.instantiate()'
julia --project=lib/CorleoneInverse/examples lib/CorleoneInverse/examples/lotka_plain/main.jl
```

The [parameter estimation guide](@ref inverse_index) starts with a one-parameter
fit and shows how to warm-start a fit when adding an experiment.

## CorleoneOED

CorleoneOED adds optimal experimental design to Corleone's
[shooting workflow](shooting.md). Use it to design controls and measurement
schedules for local parameter estimation.

Start with the [CorleoneOED overview](oed/index.md) for installation and the
[worked design tutorials](gallery/index.md). Its guides cover
[measurement conventions](oed/formulation.md), [design constraints](oed/designs.md),
and [numerical scope](oed/numerics.md).
