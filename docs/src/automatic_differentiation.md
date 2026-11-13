# [Automatic differentiation](@id automatic_differentiation)

Corleone's layers can be differentiated with several AD backends. Whether a
particular setup works depends on two choices: the outer AD backend, and the
sensitivity algorithm used inside the SciML solve. A forward solve working is not
evidence that its gradient does.

## Supported configurations

The test suite checks these configurations on full ODE solves, including losses on
saved states and continuity residuals:

| Backend | Serial | Threaded layer | Distributed layer |
|---|---|---|---|
| ForwardDiff | ✓ | ✓ | ✓ (with a named objective) |
| ReverseDiff | ✓ (pass-through sensitivities, see below) | — | — |
| Zygote | ✓ (default sensitivities) | ✓ (reverse pass runs serially) | — |

Configurations outside the table may work but are not tested. That includes
compiled ReverseDiff tapes, GPU solves, DAEs, SDEs, higher-order derivatives other
than the [sparse Hessians](@ref sparse_derivatives), and derivatives with respect to
switching or reset times. Earlier experiments with Mooncake and Enzyme are recorded
in the [validation archive](automatic_differentiation_history.md).

**Recommendation:** start with ForwardDiff. For problems with many parameters, try
Zygote.

## ForwardDiff

ForwardDiff works with the default solver settings. Reconstruct the structured
parameters from the vector *inside* the differentiated function, so that they
carry dual numbers:

```@example forwarddiff
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random, ComponentArrays, ForwardDiff

problem = ODEProblem((x, p, t) -> [p[1] * x[1]], [1.0], (0.0, 2.0), [-1.0];
    abstol = 1e-10, reltol = 1e-10)
layer = ShootingLayer(problem, Tsit5())
ps, st = LuxCore.setup(MersenneTwister(42), layer)
q0, reconstruct = pack(default_vectorizer(), ps)
loss(q) = begin
    trajectory, _ = layer(nothing, reconstruct(q), st)
    last(trajectory.u)[1]^2 + sum(abs2, shooting_constraints(trajectory))
end
gradient = ForwardDiff.gradient(loss, q0)
@assert isapprox(only(gradient), 4exp(-4); rtol = 1e-7) # hide
gradient
```

Fixed initial states are promoted to dual numbers automatically.
`OptimizationProblem(model, AutoForwardDiff())` does all of this for you; see
[objectives and constraints](@ref optimization_guide).

## ReverseDiff

ReverseDiff needs a fully specialized ODE function and pass-through sensitivities,
which let ReverseDiff trace the solver directly:

```julia
using SciMLBase, SciMLSensitivity, ReverseDiff

rhs!(dx, x, p, t) = (dx[1] = p[1] * x[1]; nothing)
f = ODEFunction{true, SciMLBase.FullSpecialize}(rhs!)
problem = ODEProblem(f, [1.0], (0.0, 2.0), [-1.0];
    sensealg = SciMLSensitivity.SensitivityADPassThrough(),
    abstol = 1e-10, reltol = 1e-10)
```

Build the shooting layer from this problem as usual. Memory use and run time differ
from an adjoint method. The tests use uncompiled tapes.

!!! warning "The default ReverseDiff sensitivity does not work"
    With SciMLSensitivity's default route, the solve returns a plain array instead
    of a solution object, and the shooting layer needs the saved times and solver
    metadata of the full solution. Use pass-through sensitivities as above. The
    [archive](automatic_differentiation_history.md#ReverseDiff) records the details.

## Zygote

Zygote works with the default SciMLSensitivity path. Corleone provides the rules it
needs for its own trajectory assembly, but not for mutation in your callbacks: use
allocating objective and constraint functions. For threaded layers, the forward
solve runs in parallel but the reverse pass runs serially.

## What is differentiated

Everything numerical that you pass in is differentiable, including values marked
`Constant`: "constant" means "not an optimization variable", not "detached from
AD". Grids, sample selection, and symbolic metadata are discrete and are not
differentiated, so derivatives hold for a fixed schedule. Don't mutate the
trajectory's data between the forward and the reverse pass; see
[trajectories](@ref trajectory_ownership).

## [Validation sources](@id ad_validation)

`test/README.md` lists the maintained test groups. Reproduction commands and dated
measurements are in the
[validation archive](automatic_differentiation_history.md#Reproducing-checks-and-measurements).
