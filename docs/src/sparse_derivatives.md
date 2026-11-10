# Preparing sparse derivatives

For ODE shooting problems on fixed schedules, structural patterns can reduce the
number of differentiation directions. Load DifferentiationInterface and
SparseMatrixColorings and enable preparation explicitly:

```julia
using Corleone, OrdinaryDiffEqTsit5, ComponentArrays, ForwardDiff
using Optimization, OptimizationMOI, Ipopt
import DifferentiationInterface, SparseMatrixColorings

problem = ODEProblem((x, p, t) -> [p[1]], [0.0], (0.0, 2.0), [0.2])
layer = ShootingLayer(problem, Tsit5(),
    Tunable(1; default = 0.2) => [1.0],
    Tunable(1; kind = Val(:variable), default = 0.2) => [1.0])
model = CorleoneDynamicOptProblem(layer, :((u₁(2.0) - 1)^2),
    :([u₁(0.5), u₁(1.5)]); lcons = zeros(2), ucons = ones(2))
optprob = OptimizationProblem(model, AutoForwardDiff();
    sparse_constraints = true, sparse_hessian = true)
solution = solve(optprob, Ipopt.Optimizer())
```

The two options are independent. `sparse_constraints=true` prepares a colored
constraint Jacobian and retains the selected AD backend for the objective.
`sparse_hessian=true` prepares exact objective and Lagrangian Hessians. Both require
`inplace=true`; do not also supply custom callbacks, prototypes, or color vectors
for the derivatives they prepare.

## Sparse constraint Jacobians

Inspect the Boolean pattern with `constraint_jacobian_sparsity(model)`. Columns
follow the parameter vectorizer; rows follow user constraints, then shooting
equalities. Patterns are constructed from model dependencies rather than by
tracing the adaptive solver.

Parsed expressions, including those lowered by ModelingToolkit, track referenced
signals and times. Literal vectors retain separate dependencies per component;
other vector expressions share their combined dependencies. Arbitrary callback
rows are dense. State dependencies propagate conservatively through integration
and partial resets; fixed resets replace them. Inserted-parameter equalities have
two entries each. Fixed numerical values in Lux state are excluded from columns.
ODE callbacks, initialization metadata, and nonidentity mass matrices keep
segment-start dependencies conservative.

Dense callback rows can limit coloring's benefit. See
[`constraint_jacobian_sparsity`](@ref) for the structural rules.

## Lagrangian Hessian structure

For packed parameters ``q``, objective factor ``\sigma``, and constraint multipliers
``\lambda``, the Lagrangian Hessian is

```math
\nabla_q^2 L(q,\sigma,\lambda)
=\sigma\nabla_q^2 f(q)+\sum_i\lambda_i\nabla_q^2 c_i(q).
```

`lagrangian_hessian_sparsity(model)` returns a symmetric Boolean pattern. With
OptimizationBase loaded, `OptimizationBase.lag_hess_structure(model)` returns
upper-triangle coordinates in CSC column order. These structural queries include
possible curvature independently of current multipliers and do not enable
numerical Hessian evaluation themselves.

Expression patterns retain all pairs within each output's dependency set. State
reads include possible nonlinear dependence through the ODE solution, even for a
linear expression in states. Matching residuals include incoming-solution curvature;
inserted parameter equalities are affine. Arbitrary callbacks use a dense fallback.
Patterns are conservative supersets and do not detect algebraic cancellation.

## Sparse exact Hessians

With `sparse_hessian=true`, objective and constraints share one prediction per
Lagrangian evaluation. Prepared callbacks use current objective factors,
multipliers, and Lux state. This requires a backend supporting second derivatives
of the model; ForwardDiff is covered by integration tests. First-order support
does not establish second-order support.

!!! note "Ipopt Hessian choice"
    `hessian_approximation="limited-memory"` does not use exact Hessian callbacks.
    Measure total solve time before choosing an exact or approximate Hessian.

## Reusing preparation

Rebuild after changing schedules, scalar/container types, or parameter/output
layouts. Numerical values may change with the same types and layout. Prepared
functions own mutable differentiation buffers: concurrent evaluations need separate
preparations and Lux states. DAE problems and other Lux layer types are outside
this structural pattern generator's scope.
