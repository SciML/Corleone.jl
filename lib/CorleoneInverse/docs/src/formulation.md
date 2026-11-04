# [Problem setup and data](@id pe_formulation)

## Statistical model

The measured data is modelled as independent Gaussian observations

```math
y_{ik} = g_k\!\big(x(t_i;\theta), \theta, t_i\big) + \varepsilon_{ik},
\qquad \varepsilon_{ik}\sim\mathcal N\!\big(0,\sigma_{ik}^2\big),
```

where ``x`` solves the model ``\dot x=f(x,\theta,t)``, ``g`` is the observation map stored in
the [`ParameterEstimationLayer`](@ref), and ``\theta`` collects the tunable initial
conditions and model parameters. The negative log-likelihood is therefore a weighted
least-squares residual, and [`problem`](@ref) returns the corresponding
`NonlinearLeastSquaresProblem`. With known ``\sigma`` the objective is

```math
\min_\theta\ \sum_{i,k}\left(\frac{y_{ik}-g_k(x(t_i;\theta))}{\sigma_{ik}}\right)^2 .
```

## Experiments and data containers

[`ExperimentData`](@ref) owns one or more [`Experiment`](@ref) objects. Each experiment
stores observation times and values either in shared-grid matrix form ``y \in
\mathbb R^{n_\text{obs}\times n_t}`` or in per-quantity vector-of-vectors form when the
measured quantities use different time grids ([`per_quantity`](@ref),
[`observation_times_per_quantity`](@ref)). The measurement noise ``\sigma`` may be a
scalar, a per-quantity/per-observation vector, or a matrix. [`n_observations`](@ref) and
[`observation_times`](@ref) provide the sizes used to assemble the residual.

Different experiments share the same model and the same tunable parameters, but each may
carry its own initial conditions and its own fixed controls
([`initial_conditions`](@ref), [`controls`](@ref), [`has_controls`](@ref)). Every
experiment is simulated with its own shooting-layer parameter tree so that local shooting
states and control schedules do not interfere.

## Shooting layers

[`ParameterEstimationLayer`](@ref) wraps a Corleone shooting layer built from the model
`ODEProblem`. The simplest constructor is

```julia
pe = ParameterEstimationLayer(prob, Tsit5(), data)
```

which builds a `ShootingLayer` with `saveat` set so that every observation time is hit.
A pre-built layer may also be supplied. Multiple shooting is available by constructing the
layer with shooting points; the flat optimization vector then additionally contains the
local shooting initial conditions, and the continuity conditions appear as residuals from
[`residual`](@ref) / [`residuals!`](@ref). [`n_shooting_constraints`](@ref) gives their
number.

Fixed parameters are represented by coincident lower and upper bounds. Such parameters are
not part of the optimization vector and stay at their supplied values. Fixed controls are
prescribed parameter histories and are likewise excluded from the optimization variables.

## Observation times and model evaluation

[`model_values`](@ref) evaluates the observation map on a trajectory. A least-squares
solver sees a non-finite residual whenever a shooting sub-interval that diverged mid-solve
does not reach an observation time; only a trajectory that contains none of the requested
times raises an error.
