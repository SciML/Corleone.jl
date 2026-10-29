# CorleoneOED

Local optimal experimental design for Corleone's shooting layers. Choose physical
controls and measurement schedules to improve information about selected model
parameters. Physical parameters not explicitly annotated are held at their nominal
values; `params` selects estimated parameters independently of design variables.

## Continuous and discrete measurements

```julia
using CorleoneOED, OrdinaryDiffEqTsit5, LuxCore, Random

problem = ODEProblem((x, p, t) -> p[1] .* x, [1.0], (0.0, 1.0), [-2.0])
design = OEDLayer(problem, Tsit5(); params = [1], measurements = (
    ContinuousMeasurement(:density, [0.0, 0.5], (x, p, t) -> x[1]),
    DiscreteMeasurement(:sample, [0.0, 0.5, 1.0], (x, p, t) -> x[1]),
))
ps, st = LuxCore.setup(Xoshiro(1), design)
trajectory, next_st = design(nothing, ps, st)
F = fisher_information(design, trajectory)
budgets = sampling_sums(design, trajectory) # [1.0, 3.0]
```

A measurement has an identifier, a grid, and an observation function `(x, p, t)`.
Scalar and vector outputs are supported. All rows of a vector-valued observation
share one weight. Weights are tunable in `[0, 1]` and enter information **linearly**.

A continuous grid specifies weight changes: start/end points are optional, and
there is one value per resulting interval. An empty grid gives a constant density.
A discrete grid specifies the actual events: one weight per time, including events
at the initial or final time. Discrete events are added to the save grid. Grids
must be sorted, unique, and contained in the problem span.

For dynamics `x′ = f(x, θ, t)` and observations `h`, the augmented system integrates

```math
G' = f_xG + f_\theta B, \qquad G(0)=0, \qquad
J = h_xG + h_\theta B.
```

Without reduction, `B` is the identity on `params`. Information combines continuous
and discrete contributions with the prior:

```math
F = F_{\mathrm{init}} + \sum_m\int w_m(t)J_m(t)^\top J_m(t)\,dt
    + \sum_d\sum_{t\in T_d}w_d(t)J_d(t)^\top J_d(t).
```

These formulas assume unit observation-noise covariance. Whiten observation
outputs yourself to represent a known nonidentity covariance. Continuous sampling
budgets integrate weight over time; discrete budgets sum event weights.

## Optimize a design

```julia
using Optimization, ComponentArrays, OptimizationMOI, Ipopt
model = CorleoneDynamicOptProblem(design, ACriterion(); ps, st, M=[0.2, 1.0])
optprob = OptimizationProblem(model, AutoForwardDiff())
result = solve(optprob, Ipopt.Optimizer(); hessian_approximation="limited-memory")
_, reconstruct = pack(default_vectorizer(), ps)
optimal_trajectory, _ = design(nothing, reconstruct(result.u), st)
```

The convenience constructor `OptimizationProblem(design, ACriterion(); ps, st, M)`
also supports `integer_weights=true`, which marks measurement weights only. Both
forms use the core SciMLBase bridge; no OED Optimization extension is required.
Integrality masks use the same vectorizer as parameters and bounds. Choose
an integer-capable solver for that option. The relaxed problem works with Ipopt.

Bounds use `get_bounds(design, ps, st)`. Sampling-budget bounds `M` are ordered
continuous then discrete. Extra constraints use the core facade's callback or
expression interface and structured `lcons`/`ucons`; automatic shooting equalities
are appended last. A/D/E criteria minimize trace, determinant, or largest eigenvalue
of inverse information; `Fisher*Criterion` variants maximize the corresponding
information measures. Inverse-based criteria require nonsingular information.

## Shooting, repeated experiments, and priors

Pass ordinary `Tunable(...) => grid` definitions for controls and lifted states,
or construct `OEDLayer(existing_shooting_layer; params, measurements)`. Mark physical
controls with `controls=[indices...]` for symbolic bookkeeping and plotting.
Full physical-state shooting resets also lift the sensitivity/information states.
Their matching equalities are essential: arbitrary initial guesses need not yield
a positive-definite terminal information matrix. Initialize the augmented shooting
layer with `forward_initialization(rng, design.shooting, ps, st)` when needed.
Prescribed physical resets after the initial time, including a final-time reset,
require sensitivity reset maps and are rejected. Free initial values can be design variables, but estimated
initial-state sensitivities are not included in `params`.

`MultiExperimentLayer((design1, design2))` combines independent design parameter
and state trees. Local matrices are embedded by physical parameter index and
summed. The same index must denote the same physical quantity in every experiment.
Use `MultiExperimentLayer(design, n)` for repeated copies with independent Lux trees.

`update_fim(design, [(; ps, st)], state)` adds completed experiments' information to
the prior without counting their existing priors again. Multi-experiment models
have one global prior; child priors are excluded from their aggregate.

## Numerical scope

Augmentation traces the RHS and observations with Symbolics, so those functions
must accept symbolic inputs. The current interface expects vector states and
vector physical parameters. It supports ODEs with constant mass matrices and
residual-form DAEs; consistent initial conditions and a suitable solver are still
required. It does not construct a ModelingToolkit initialization system.

`svd=true` derives a fixed reduced parameter-direction basis from a nominal solve;
`ns` selects its dimension. Singular-value and vector thresholds control retained
directions/rows. The reference matrix is a time-integrated nominal sensitivity
surrogate, not the optimized discrete design. For multiple experiments, reduced
information is projected back through each fixed basis before summation; this is
a reduced approximation, not recovery of discarded information.

ForwardDiff differentiates the numerical design solve. Other backends depend on
the underlying shooting layer, solver, and sensitivity algorithm; do not infer
backend support merely from successful symbolic augmentation. See the root AD
backend documentation for its solver-specific restrictions.

Trajectories save augmented **states** only. Physical controls and measurement
weights remain parameter histories, queried through SymbolicIndexingInterface.
Sampling budgets can also be evaluated directly from `(design, nothing, ps, st)`
without an ODE solve.

## API reference

The standalone Documenter site lives in `docs/`. Develop the checked-out root
package and CorleoneOED in its environment, then run `julia --project=docs docs/make.jl`
from this package directory.

## Attribution

The measurement abstraction, mixed continuous/discrete design, multi-experiment
work, and DAE augmentation were adapted from **Christoph Plate (@chplate)** in
[AlCap23/Corleone.jl](https://github.com/AlCap23/Corleone.jl), using its `main`
branch at [7443580115](https://github.com/AlCap23/Corleone.jl/commit/7443580115fa432e4996a67b45a9028b3d10e7ed).
See also [the measurement abstraction](https://github.com/AlCap23/Corleone.jl/commit/4d300d3337)
and [multi-experiment work](https://github.com/AlCap23/Corleone.jl/commit/70d8b51e3a).
This port adapts those contributions to the rewritten Corleone API. The original
MIT license and author credits are retained in [LICENSE](LICENSE) and `Project.toml`.
