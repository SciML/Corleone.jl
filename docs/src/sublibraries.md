# Sublibraries

Corleone's core package builds and optimizes shooting problems. Two sublibraries
build on it for the two most common follow-up questions: *which parameters explain
my data?* and *which experiment should I run next?*

## CorleoneInverse

[CorleoneInverse](@ref inverse_index) fits model parameters to measurements. It
handles several experiments with shared and experiment-specific values, priors and
positivity transforms, and reports covariances and confidence intervals. Start with
the [first fit](@ref inverse_index) or the
[Lotka–Volterra tutorial](@ref inverse_lotka_plain).

## CorleoneOED

[CorleoneOED](@ref oed_overview) designs experiments for local parameter
estimation. It optimizes controls and measurement schedules to maximize the Fisher
information of chosen parameters. Start with the [overview](@ref oed_overview) or the
[Lotka–Volterra design tutorial](@ref lotka_oed).

## Running the sublibrary examples

Each sublibrary keeps its example scripts in `lib/<Name>/examples`. To run one from
a repository checkout, set up the example environment once and include the script,
for example:

```sh
julia --project=lib/CorleoneInverse/examples -e 'using Pkg; Pkg.develop([Pkg.PackageSpec(path="."), Pkg.PackageSpec(path="lib/CorleoneInverse"), Pkg.PackageSpec(path="lib/CorleoneOED")]); Pkg.instantiate()'
julia --project=lib/CorleoneInverse/examples lib/CorleoneInverse/examples/lotka_plain/main.jl
```
