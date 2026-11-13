# [Sparse derivatives](@id sparse_derivatives)

In a shooting problem, most constraints depend on only a few variables: a sampled
state depends on the controls before it, and a continuity residual on its two
neighboring segments. Corleone derives these sparsity patterns from the problem
structure, and with them, colored AD needs far fewer directions than a dense
Jacobian or Hessian.

## Enabling sparse derivatives

Load DifferentiationInterface and SparseMatrixColorings, then request sparse
preparation when you build the optimization problem:

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

The two options are independent:

- `sparse_constraints = true` prepares a colored constraint Jacobian. The objective
  gradient still uses the AD backend you chose.
- `sparse_hessian = true` prepares exact Hessians of the objective and the
  Lagrangian. Objective and constraints then share one solve per evaluation.

Both require `inplace = true`, the default. Don't also pass your own derivative
functions, prototypes, or colorings for the parts they prepare.

!!! note "Ipopt and exact Hessians"
    With `hessian_approximation = "limited-memory"`, Ipopt never calls the exact
    Hessian. Whether exact or approximate Hessians are faster depends on the
    problem, so measure the total solve time.

Exact Hessians need a backend that supports second derivatives of the model;
ForwardDiff is covered by the integration tests.

## Inspecting the patterns

`constraint_jacobian_sparsity(model)` returns the Boolean Jacobian pattern, with
columns in the order of the [packed parameters](@ref parameter_packing) and rows for
your constraints followed by the shooting equalities.
`lagrangian_hessian_sparsity(model)` returns the symmetric pattern of

```math
\nabla_q^2 L(q,\sigma,\lambda)
=\sigma\nabla_q^2 f(q)+\sum_i\lambda_i\nabla_q^2 c_i(q),
```

for the packed parameters ``q``, objective factor ``\sigma``, and multipliers
``\lambda``. With OptimizationBase loaded, `OptimizationBase.lag_hess_structure(model)`
returns its upper triangle as coordinates in CSC order. These queries only describe
structure; they don't evaluate anything.

The patterns are conservative: they may contain entries that turn out to be zero,
but never miss a nonzero. How sparse they are depends on how you write the model:

- **Expressions** are analyzed: Corleone tracks which signals and times each
  output reads. The entries of a literal vector, such as `:([u₁(0.5), u₁(1.5)])`,
  get separate rows; other vector expressions share one combined pattern.
- **Callbacks** are opaque and get dense rows, which can remove most of the
  benefit.
- **State reads** depend on all earlier controls and shooting values of their
  segment, possibly nonlinearly. Fixed resets cut this dependence; ODE callbacks,
  initialization metadata, and non-identity mass matrices keep it conservative.
- **Inserted-parameter equalities** are affine, with two entries each.

The [`constraint_jacobian_sparsity`](@ref) docstring lists the exact rules.

## Reusing preparation

Prepared derivatives can be reused as long as the schedules, the scalar and
container types, and the parameter and output layouts stay the same; only numerical
values may change. They own mutable buffers, so concurrent evaluations need separate
preparations and Lux states. The pattern generator covers ODE shooting layers;
DAEs and other Lux layers are outside its scope.
