# CorleoneOED

Local optimal experimental design for Corleone's shooting layers. Choose physical
controls and measurement schedules to improve information about selected physical
parameters at nominal values. Continuous and discrete measurements can coexist;
scalar and vector observations share one weight per measurement.

```julia
using CorleoneOED, OrdinaryDiffEqTsit5, LuxCore, Random

problem = ODEProblem((x, p, t) -> p[1] .* x, [1.0], (0.0, 1.0), [-2.0])
design = OEDLayer(problem, Tsit5(); params=[1], measurements=(
    ContinuousMeasurement(:density, [0.0, 0.5, 1.0], (x, p, t) -> x[1]),
    DiscreteMeasurement(:sample, [0.0, 0.5, 1.0], (x, p, t) -> x[1]),
))
ps, st = LuxCore.setup(Xoshiro(1), design)
trajectory, next_st = design(nothing, ps, st)
F = fisher_information(design, trajectory)
budgets = sampling_sums(design, trajectory) # [1.0, 3.0]
```

Continuous grids define density changes: endpoints add no values, so this example
has two density intervals. Discrete grids define events: it has three event weights,
including both endpoints. Information is linear in weights. `params` selects
estimated physical parameters independently of their tunability; unannotated
physical parameters remain fixed under the constructor's default policy.

## Documentation and executable tutorials

The standalone [manual](docs/src/index.md) separates the
[mathematical formulation](docs/src/formulation.md),
[design and optimization guide](docs/src/designs.md),
[SVD, AD, and numerical scope](docs/src/numerics.md), and [API](docs/src/api.md).
Build it with the [documentation environment and instructions](docs/README.md).
The [root Corleone manual](../../docs/README.md) also includes these tutorials and
the same guide/API sources. Both builds execute the authoritative Literate sources:

- [Lotka–Volterra](examples/lotka_oed/main.jl): fishing and continuous measurement densities.
- [Multiple experiments](examples/lotka_multiexperiments/main.jl): independent schedules and global prior accounting.
- [Compartmental model](examples/compartmental_oed/main.jl): discrete candidate events and relaxed sampling decisions.

The quick start needs no Optimization or ComponentArrays import. Tutorials add
ComponentArrays for the shared `pack` vectorizer, Optimization/OptimizationMOI and
Ipopt for continuous optimization, and CairoMakie for plots. There is no OED-specific
optimization extension. Integer masks mark measurement weights only and require an
integer-capable solver to enforce binary decisions; Ipopt solves the relaxation.

Trajectories retain the supplied parameter and state containers. Parameter histories
are queried on demand, so re-solve after changing those values when saved states
and parameter queries must describe the same experiment. The manual documents
shooting continuity, current priors, supported DAE formulations, and tested AD paths
without inferring AD support from a successful forward solve.

## Attribution

The measurement abstraction, mixed continuous/discrete design, multi-experiment
work, and DAE augmentation were adapted from **Christoph Plate (@chplate)** in
[AlCap23/Corleone.jl](https://github.com/AlCap23/Corleone.jl), using its `main`
branch at [7443580115](https://github.com/AlCap23/Corleone.jl/commit/7443580115fa432e4996a67b45a9028b3d10e7ed).
See also [the measurement abstraction](https://github.com/AlCap23/Corleone.jl/commit/4d300d3337)
and [multi-experiment work](https://github.com/AlCap23/Corleone.jl/commit/70d8b51e3a).
This port adapts those contributions to the rewritten Corleone API. The original
MIT license and author credits are retained in [LICENSE.md](LICENSE.md) and `Project.toml`.
