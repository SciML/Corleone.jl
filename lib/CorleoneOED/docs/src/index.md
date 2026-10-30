# [CorleoneOED](@id oed_overview)

CorleoneOED designs controls and measurement schedules for local parameter
estimation in differential-equation models. It augments a Corleone `ShootingLayer`
with state sensitivities and Fisher information. “Local” means information is
computed at nominal physical parameter values, rather than averaged over an
unknown parameter distribution.

Start with the example below, then follow a complete executable tutorial:

- [Continuous Lotka–Volterra design](@ref lotka_oed): optimize fishing and two measurement densities.
- [Multiple experiments](@ref lotka_multiexp): combine independent schedules and a global prior.
- [Discrete compartmental design](@ref compartmental_oed): allocate relaxed weights to candidate sampling events.

The [formulation](@ref oed_formulation) defines what is computed;
the [design guide](@ref oed_designs) explains parameter storage, shooting,
vectorization, constraints, and priors. [Numerical scope](@ref oed_numerics) covers
SVD reduction, DAE assumptions, and verified AD behavior. Exact signatures and
return values live in the [API reference](@ref oed_api).

## A first mixed design

This example needs CorleoneOED, LuxCore, Random, and OrdinaryDiffEqTsit5. It does
not need Optimization or ComponentArrays. The model is ``\dot x=\theta x`` at
``\theta=-2``; the nominal rate is fixed, while all measurement weights start at one.

```@example quickstart
using CorleoneOED, LuxCore, Random, OrdinaryDiffEqTsit5

problem = ODEProblem((x, p, t) -> p[1] .* x, [1.0], (0.0, 1.0), [-2.0];
    abstol=1e-10, reltol=1e-10)
design = OEDLayer(problem, Tsit5(); params=[1], measurements=(
    ContinuousMeasurement(:density, [0.0, 0.5, 1.0], (x, p, t) -> x[1]),
    DiscreteMeasurement(:sample, [0.0, 0.5, 1.0], (x, p, t) -> x[1]),
))
ps, st = LuxCore.setup(Xoshiro(1), design)
trajectory, next_st = design(nothing, ps, st)
F = fisher_information(design, trajectory)
budgets = sampling_sums(design, trajectory)
@assert size(F) == (1, 1) && only(F) > 0
@assert budgets ≈ [1.0, 3.0]
(; information=only(F), budgets)
```

The continuous grid makes **two** density intervals, not three parameter values:
its endpoints are boundaries. The discrete grid specifies **three** actual
observations, including both endpoints. Selecting `params=[1]` estimates
information about `p[1]`; it does not make `p[1]` a tunable design variable.

For comparison, the exact information here is
``(2-26e^{-4})/64 + e^{-2}/4 + e^{-4}``. The first term is continuous information;
the initial discrete observation has zero state sensitivity.

```@example quickstart
reference = (2 - 26exp(-4))/64 + exp(-2)/4 + exp(-4)
@assert isapprox(only(F), reference; rtol=1e-7)
sampling_sums(design, nothing, ps, st) # the same budgets, without an ODE solve
```

The returned trajectory retains its parameter and state containers by reference.
Re-solve after changing them when you need saved states and parameter queries to
represent the same experiment; see [ownership and histories](@ref oed_ownership).

## Next: solve a constrained design

The tutorials add ComponentArrays for packing, Optimization/OptimizationMOI and
Ipopt for the continuous optimizer, and CairoMakie for plots. Their random seeds,
solver settings, reconstruction, termination checks, and budget checks are part
of the executable sources. An integer mask is demonstrated separately from the
continuous relaxation.

For a checkout, follow the [documentation build instructions](https://github.com/SciML/Corleone.jl/blob/main/lib/CorleoneOED/docs/README.md).
Scientific references and software provenance are listed [separately](@ref oed_references).
