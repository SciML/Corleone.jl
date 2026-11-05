# [CorleoneOED](@id oed_overview)

CorleoneOED designs controls and measurement schedules for local parameter
estimation in differential-equation models. It augments a Corleone `ShootingLayer`
with state sensitivities and Fisher information, evaluated at nominal physical
parameter values. Estimated parameters and tunable design variables are distinct.

## Start here

Install `CorleoneOED` and `OrdinaryDiffEqTsit5` for layer evaluation; the worked
optimization tutorials also import their optimizer, packing, and plotting packages.
Start with [continuous Lotka–Volterra design](@ref lotka_oed) to optimize fishing
and two measurement densities. Then choose:

- [Discrete compartmental design](@ref compartmental_oed) to allocate relaxed
  weights to candidate observations.
- [Multiple experiments](@ref lotka_multiexp) to combine independent designs and a
  global prior.

## Guides and reference

[Measurements and information](@ref oed_formulation) defines grids, sensitivities,
noise conventions, budgets, and criteria. [Constructing and optimizing designs](@ref oed_designs)
covers controls, augmented shooting, constraints, and priors.
[Numerical scope](@ref oed_numerics) explains SVD reduction, DAE assumptions, and
AD support. Exact signatures are in the [API reference](@ref oed_api);
[references and attribution](@ref oed_references) record scientific and software sources.

For shared layer construction, read Corleone's
[shooting guide](https://github.com/SciML/Corleone.jl/blob/main/docs/src/shooting.md).
To build this sublibrary's manual, follow the
[build instructions](https://github.com/SciML/Corleone.jl/blob/main/lib/CorleoneOED/docs/README.md).
