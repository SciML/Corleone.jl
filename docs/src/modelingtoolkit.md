# ModelingToolkit frontend

Load ModelingToolkit to construct a `CorleoneDynamicOptProblem` from an uncompiled
`System`, its initial values, and control schedules:

```julia
using Corleone, ModelingToolkit, OrdinaryDiffEqTsit5
using ComponentArrays, Optimization
using ModelingToolkit: t_nounits as t, D_nounits as D

@variables x(..) = 1.0 [tunable = false]
@variables u(..) = 0.0 [input = true, bounds = (-1.0, 1.0)]
@parameters a = 1.0 [tunable = false]
I = ModelingToolkit.Symbolics.Integral(t in (0.0, 2.0))

@named sys = System(
    [D(x(t)) ~ -a*x(t) + u(t)], t;
    costs = [I(x(t)^2 + u(t)^2)],
    constraints = [x(2.0) ≲ 0.5],
)

model = CorleoneDynamicOptProblem(
    sys, [], u(t) => 0.0:0.25:1.75;
    algorithm = Tsit5(),
)
optprob = OptimizationProblem(model, AutoForwardDiff())
```

The frontend sums the system's costs, replaces definite time integrals with
quadrature states, and translates symbolic constraints into residuals and bounds.
Equations have zero lower and upper bounds; inequalities use a residual bounded
above by zero. Saved values at reset times select the post-reset state, as in the
numerical expression API. Bare parameters refer to their terminal values.

`Annotations(sys)` reads MTK metadata directly and returns
`(; variables, parameters)`. Each entry is a `Tunable` or `Constant` annotation
with its index, symbol, kind, numerical default, bounds and integrality specified.
Use `Annotations(sys; u0, p)` to apply numerical problem values instead of symbol
defaults. The frontend uses this constructor before assigning control and shooting
schedules; fixed parameters are represented by `Constant` annotations.

Controls remain parameters. A grid beginning at `tspan[1]` describes interval
starts, preserving the old MTK calling convention. That first point is removed
before constructing the native switching grid. Grids containing only interior
switches are also accepted. Bounds come from symbol metadata; parameters marked
`tunable = false` are fixed, except for explicitly selected controls.

State metadata such as `x(..) = 0.5 [tunable = true, bounds = (0.3, 1.0)]`
makes the value at `tspan[1]` trainable, initialized to `0.5` and bounded by
`0.3` and `1.0`. This value replaces the problem's initial state freely; no
matching equality pins it to the original default. States without this metadata
retain their numerical initial values.

The time span is inferred from integral endpoints, timed reads, and supplied
grids. Pass `tspan = (start, stop)` to set it explicitly. All requested times must
lie within that span. Pass solver options such as tolerances through
`problem_kwargs = (; abstol = 1e-8, reltol = 1e-8)`.
The frontend disables MTK initialization-system construction. Supply numerical
initial values or defaults for the ODE states and parameters.

For multiple shooting, add `shooting = [0.5, 1.0, 1.5]`. Every numerical state,
including quadratures, receives tunable reset values at interior shooting times.
Initial values are free only for states marked `tunable = true`; other states
start from the ODE problem's values. The numerical facade appends continuity
equalities at interior resets automatically. `ensemblealg` selects the execution
method. Controls and physical parameters remain shared between segments.

The system is compiled with `split = false`, retaining MTK's symbolic container
and using flat numerical parameters. This frontend currently expects scalar
signals that survive compilation as numerical states or parameters. Array-valued
parameters are rejected: MTK's flat-parameter problem constructor does not support
them. Declare separate scalar parameters for this frontend. Costs and
constraints must use numerical times; references to eliminated signals are
rejected. The `evaluator` keyword selects the same expression compiler as the
numerical facade.
