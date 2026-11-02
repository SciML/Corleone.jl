# ModelingToolkit frontend

See the [ModelingToolkit Lotka–Volterra tutorial](examples/modelingtoolkit_integration.md)
for a complete single- and multiple-shooting example. The
[LTI tutorial](examples/lti_control_with_unknown_coefficients.md) jointly optimizes
a scalar input and three constant coefficients under terminal state equalities.

## Define the model

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

## Annotations and metadata

`Annotations(sys)` reads MTK metadata directly and returns
`(; variables, parameters)`. Each entry is a `Tunable` or `Constant` annotation
with its index, symbol, kind, numerical default, bounds and integrality specified.
Use `Annotations(sys; u0, p)` to apply numerical problem values instead of symbol
defaults. The frontend uses this constructor before assigning control and shooting
schedules; fixed parameters are represented by `Constant` annotations.

## Control schedules

Controls remain parameters. A grid beginning at `tspan[1]` describes interval
starts, preserving the old MTK calling convention. That first point is removed
before constructing the native switching grid. Grids containing only interior
switches are also accepted. Bounds come from symbol metadata; parameters marked
`tunable = false` are fixed, except for explicitly selected controls.

## Initial values and time span

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

## Multiple shooting and state bounds

For multiple shooting, add `shooting = [0.5, 1.0, 1.5]`. Every numerical state,
including quadratures, receives tunable reset values at interior shooting times.
Initial values are free only for states marked `tunable = true`; other states
start from the ODE problem's values. The numerical facade appends continuity
equalities at interior resets automatically. `ensemblealg` selects the execution
method. Controls and physical parameters remain shared between segments.

!!! note "State bounds constrain stored shooting values"
    Bounds apply to lifted values at shooting times, including states marked
    `tunable = false`. That metadata fixes the global initial value; it does not
    disable interior shooting variables or their bounds. With no free initial
    value or lifted states, state metadata adds no optimizer variable bounds.
    Add explicit sampled constraints when state limits must also be checked away
    from shooting nodes. Neither those samples nor shooting bounds certify a
    continuous path constraint between samples.

## Array parameters and expression evaluation

The system is compiled with `split = false` and uses flat numerical parameters.
Array parameters are expanded into indexed scalar symbols during setup, in
column-major order. Their defaults, bounds and tunability are preserved; each
component keeps its own scalar vector buffer and components on the same grid
share a lookup. Original component symbols such as `a[1]` remain usable for
trajectory queries and symbolic costs or constraints. Timed state/control reads
still require scalar signals. References to eliminated signals are rejected.
The `evaluator` keyword selects the same expression compiler as the numerical facade.
