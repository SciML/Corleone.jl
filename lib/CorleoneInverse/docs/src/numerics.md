# [Fitting, transforms, and regularization](@id pe_numerics)

## Solver interface

[`problem`](@ref) returns a `ParameterEstimationProblem`, an
`SciMLBase.AbstractNonlinearProblem` whose residual maps the flat optimization vector to the
weighted observation residuals (plus any shooting-continuity and regularization residuals).
It can be passed directly to a NonlinearSolve least-squares solver:

```julia
sol = solve(problem(pe), GaussNewton(); abstol = 1e-6, reltol = 1e-6)
est = estimate_parameters(pe, sol)
```

[`estimate_parameters`](@ref) unpacks the solution into a [`ParameterEstimate`](@ref)
holding the point estimate, the covariance, standard errors, and confidence intervals. The
accessors [`covariance`](@ref), [`standard_errors`](@ref), [`confidence_interval`](@ref),
[`r_squared`](@ref), [`adjusted_r_squared`](@ref), [`correlation`](@ref), and
[`condition_number`](@ref) expose the same diagnostics. The covariance is formed from the
residual Jacobian at the optimum; inversion of the normal matrix uses a Cholesky
factorization with a pseudo-inverse fallback, so rank-deficient parameter combinations are
reported instead of crashing.

## Parameter transforms

By default the optimizer works on the natural parameters. The `transforms` keyword instead
fits an unconstrained internal variable ``\phi`` for every entry of the shared ``[u_0;p]``
block, with the natural value recovered as ``\theta=\mathrm{from\_internal}(t,\phi)``.
[`Identity`](@ref) is the default, while [`Affine`](@ref), [`Log`](@ref), and [`Logit`](@ref)
enforce ranges on the natural parameters. Local multiple-shooting states are never
transformed. The flat vector accepted and returned by the problem is always expressed in
the internal space.

## Regularization

The `regularization` keyword appends a data-independent penalty on the fitted model
parameters to the objective. [`L2Regularizer`](@ref), [`L1Regularizer`](@ref), and
[`ElasticNetRegularizer`](@ref) cover ridge, lasso, and elastic-net penalties. Because the
regularization residuals are appended after the observation residuals, the reported
uncertainties still refer to the data part of the objective.

## Profile likelihood

[`profile_likelihood`](@ref) computes a likelihood-ratio profile for selected parameters by
re-minimizing the residual sum of squares on a grid around the optimum. The returned
[`ProfileLikelihoodEstimate`](@ref) exposes one [`ProfileCurve`](@ref) per parameter
(parameter values, residual sums of squares, minimum, threshold, and the profile confidence
interval) together with the asymptotic intervals for cross-checking.
