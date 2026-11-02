# [Measurements and Fisher information](@id oed_formulation)

## Parameters, directions, and design variables

Write the physical model as

```math
\dot x=f(x,\theta,v,t),\qquad x(t_0)=x_0,
```

where ``x\in\mathbb R^{n_x}``, ``\theta\in\mathbb R^{n_\theta}`` are the
selected physical parameters, and ``v`` denotes physical controls or other design
choices. In Julia, both physical parameters and controls occupy entries of the
original vector `problem.p` of length ``n_p``; `params` selects the indices of ``\theta`` in the
requested order. `Tunable`/`Constant` annotations independently determine which
values can vary during design optimization. Unannotated physical parameters are
fixed at nominal values under the OED constructor's default `fixed_parameters` policy.

The matrix ``B\in\mathbb R^{n_\theta\times n_s}`` defines parameter directions.
With no reduction, ``n_s=n_\theta`` and ``B=I`` on the selected coordinates.
The package integrates ``G\in\mathbb R^{n_x\times n_s}``:

```math
\dot G=f_xG+f_\theta B,\qquad G(t_0)=0.
```

For ``B=I``, ``G`` contains state sensitivities with respect to the selected
physical parameters, holding the design schedule fixed. The column count is set
by these selected coordinates and the basis, not by the number of scheduled
trainable values. The reduced case is
explained in [SVD reduction](@ref oed_svd). Initial conditions are assumed independent
of these estimated parameters. Free initial values can be design variables, but
`params` does not select estimated initial conditions or initialize their sensitivities.

These sensitivity columns are distinct from the AD gradient of a design objective
with respect to optimizer variables. AD differentiates *through* the augmented
numerical solve, including its sensitivities and information, as the design changes.

## [Measurement grids and outputs](@id oed_measurements)

A measurement is `ContinuousMeasurement(id, tpoints, observed)` or
`DiscreteMeasurement(id, tpoints, observed)`, with `observed(x, p, t)` returning a
scalar or vector. The observation map must accept symbolic inputs and keep its
output dimension fixed. A vector observation has one shared weight for all its
output rows; use separate measurements for independently weighted outputs.

Identifiers must be unique and distinct from model symbols. Grids are validated
when constructing the OED layer: times must be finite, sorted, unique, and inside
`problem.tspan`. A discrete grid must be nonempty; an empty continuous grid is valid.

| Grid on `(0.0, 1.0)` | Stored weights | Meaning |
|---|---:|---|
| Continuous `[]` | 1 | Constant density over the full span |
| Continuous `[0.5]` | 2 | Densities on the two intervals separated at `0.5` |
| Continuous `[0.0, 0.5, 1.0]` | 2 | The same two intervals; endpoints add no density values |
| Discrete `[0.0, 0.5, 1.0]` | 3 | Three observation events, each with its own weight |

Weights are tunable in ``[0,1]`` and initially one. Continuous densities are
piecewise constant, taking the outgoing interval's value at a switch. Discrete
weights are also represented by a right-continuous parameter schedule, but only
the values at the declared events contribute information or discrete budgets.
The first event's weight is stored on the initial interval even if the first event
is later than ``t_0``. There is no extra discrete sample before that event.

Discrete event times are added to the saved grid. At a shooting reset, both
incoming and outgoing states are retained, and discrete information selects the
**last saved sample at that time**, the outgoing/post-reset state. General
`observed_equations` and `sensitivities` outputs include both saved sides.
Parameter histories are right-continuous on either saved side; see the
[Corleone trajectory guide](https://github.com/SciML/Corleone.jl/blob/main/docs/src/trajectories.md).

## Observation sensitivities and information

For measurement ``m`` with ``r_m`` output rows, define
``h_m:\mathbb R^{n_x}\times\mathbb R^{n_p}\times\mathbb R\to\mathbb R^{r_m}``.
The implemented observation-sensitivity matrix has size ``r_m\times n_s``:

```math
J_m=h_{m,x}G+h_{m,\theta}B.
```

Direct dependence on estimated parameters matters. An observation such as
`(x, p, t) -> p[1]` can provide information even if its state derivative is zero.
The supplied map's rows are all included; no normalization by output count is applied.

Let ``\mathcal C`` and ``\mathcal D`` be continuous and discrete measurements,
and ``T_m`` the event grid of a discrete measurement. Terminal information is

```math
F=F_{\mathrm{prior}}
+\sum_{m\in\mathcal C}\int_{t_0}^{t_f}w_m(t)J_m(t)^\top J_m(t)\,dt
+\sum_{m\in\mathcal D}\sum_{t\in T_m}w_m(t)J_m(t)^\top J_m(t).
```

Here ``F\in\mathbb R^{n_s\times n_s}`` and `st.F_init` supplies the prior in
these coordinates. The ODE integrates the upper triangle of the continuous
contribution, initially zero. Discrete contributions are accumulated from saved
states on demand. `fisher_information(design, trajectory)` combines them with the
current prior without another differential-equation solve.

Information is **linear** in the measurement weight, ``wJ^\top J``, rather than
quadratic in that weight. This is a sampling-design weighting convention, not
multiplication of the observation by ``w`` before forming its Jacobian.

The formula assumes independent, whitened observation errors with unit covariance.
To represent a known, fixed positive-definite covariance ``R_m=L_mL_m^\top``,
supply the whitened map ``L_m^{-1}h_m``. Scalar standard deviations are handled by
dividing the observation by the standard deviation. Correlations across distinct
measurements/events and parameter-dependent noise information are not assembled
by this formula.

## Budgets and information gains

The corresponding budgets are

```math
b_m=\int_{t_0}^{t_f}w_m(t)\,dt\quad(m\in\mathcal C),\qquad
b_m=\sum_{t\in T_m}w_m(t)\quad(m\in\mathcal D).
```

Continuous budgets integrate the *current* piecewise-constant schedule exactly
by interval length; discrete budgets sum event weights. They need no additional
ODE solve. `sampling_sums` and optimizer bounds `M` order continuous measurements
first, then discrete measurements, retaining declaration order within each kind.
Multiple experiments concatenate that order experiment by experiment.

`local_information_gain` returns unweighted outer products of individual rows of
``J`` at every saved time. `global_information_gain` returns
``(J_iF^{-1})^\top(J_iF^{-1})`` for each row ``i``. For symmetric nonsingular
information, its trace gives the magnitude of the row's marginal improvement in
A-optimality before interval length/event weighting. For a vector measurement,
sum its rows to obtain its shared weight's contribution. These routines do not
select only the discrete events.

## [Criteria](@id oed_criteria)

All criteria are minimized. Their exact definitions are:

| Criterion | Scalar returned |
|---|---|
| `ACriterion()` | ``\operatorname{tr}(F^{-1})`` |
| `DCriterion()` | ``1/\det(F)`` |
| `ECriterion()` | ``\lambda_{\max}(F^{-1})`` |
| `FisherACriterion()` | ``-\operatorname{tr}(F)`` |
| `FisherDCriterion()` | ``-\det(F)`` |
| `FisherECriterion()` | ``-\lambda_{\min}(F)`` |

```jldoctest criteria
julia> using CorleoneOED, LinearAlgebra

julia> F = diagm([2.0, 4.0]);

julia> (ACriterion()(F), DCriterion()(F), ECriterion()(F))
(0.75, 0.125, 0.5)

julia> (FisherACriterion()(F), FisherDCriterion()(F), FisherECriterion()(F))
(-6.0, -8.0, -2.0)
```

These are not log-determinant or dimension-normalized criteria. Matrix inputs are
wrapped in `Symmetric`; supply symmetric information. No diagonal regularization
or positive-definiteness check is added by the criteria themselves.

!!! note "Information conditioning"
    Inverse-based criteria require invertible information. Positive-definite
    information is the relevant case for an identifiable local covariance
    interpretation. A singular or poorly conditioned design may need different
    measurements, a scientifically justified prior, or reduced coordinates.
    Eigenvalue-based criteria can be nondifferentiable at repeated extreme eigenvalues.
