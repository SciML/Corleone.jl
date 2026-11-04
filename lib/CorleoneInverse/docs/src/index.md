# [CorleoneInverse](@id pe_overview)

CorleoneInverse fits the unknown parameters of an ordinary differential
equation model to noisy time-series observations with a nonlinear least-squares solver, and
then reports parameter uncertainties from the residual Jacobian at the optimum.

It works with a Corleone `ShootingLayer` (single or multiple shooting), an observation map,
and measured data. The fitted problem is an
`SciMLBase.AbstractNonlinearProblem` built by [`problem`](@ref), so any least-squares solver
from [NonlinearSolve.jl](https://github.com/SciML/NonlinearSolve.jl) can be used. The
package is usable on its own, but is also the estimation back end of the wider
[Corleone.jl](https://github.com/SciML/Corleone.jl) optimal-control and
[CorleoneOED](https://github.com/SciML/Corleone.jl) experimental-design stack.

- Start with the [plain Lotka–Volterra tutorial](@ref corleone_pe_lotka_plain) for a single
  noisy experiment, then combine experiments in the
  [multi-experiment tutorial](@ref corleone_pe_lotka_multiexp).
- Estimate under a fixed design-derived control in the
  [fixed-control tutorial](@ref corleone_pe_lotka_oed_control) and use the
  [multiple-shooting tutorial](@ref corleone_pe_ms_lotka_oed) for long horizons.
- Fit stiff differential-algebraic systems in the [Robertson DAE tutorial](@ref corleone_pe_rober_dae),
  pool epidemic data in the [SIR tutorial](@ref corleone_pe_sir_multicity), and embed a
  neural network in the [universal differential equation tutorial](@ref corleone_pe_ude).

The [user guide](@ref pe_formulation) defines the statistical model and the
[fit guide](@ref pe_numerics) covers transforms, regularization, and uncertainty.
Exact signatures live in the [API reference](@ref pe_api).

## A first fit

This example needs CorleoneInverse, LuxCore, Random, NonlinearSolve, and
OrdinaryDiffEqTsit5. The model is ``\dot u=-p u`` at ``p=0.7``; the fit starts from
``p=0.3``.

```@example quickstart
using Corleone, CorleoneInverse, LuxCore, Random
using NonlinearSolve, OrdinaryDiffEqTsit5

f = (du, u, p, t) -> (du[1] = -p[1] * u[1]; nothing)
prob = ODEProblem(f, [1.0], (0.0, 5.0), [0.3])

tdata = collect(0.0:0.5:5.0)
ydata = exp.(-0.7 .* tdata)
data = ExperimentData(tdata, reshape(ydata, 1, :); σ = 0.01)

pe = ParameterEstimationLayer(prob, Tsit5(), data)
sol = solve(problem(pe), GaussNewton())
est = estimate_parameters(pe, sol)
@assert isapprox(est.x[1], 0.7; atol = 0.05)
@assert r_squared(est) > 0.99
nothing
```

The default construction keeps fixed parameters (coincident bounds) out of the optimization
vector and only fits the remaining tunable parameters and initial conditions.
