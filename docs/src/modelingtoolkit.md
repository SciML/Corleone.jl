# [ModelingToolkit frontend](@id modelingtoolkit_guide)

With ModelingToolkit loaded, you can describe the whole problem symbolically:
dynamics, integral costs, constraints, bounds, and which values are tunable.
Corleone then builds the shooting layer and the optimization model from the
`System`.

## Defining the model

Pass an uncompiled `System`, initial values (here none), and the control schedules
to `CorleoneDynamicOptProblem`:

```@example modelingtoolkit_guide
using Corleone, ModelingToolkit, OrdinaryDiffEqTsit5
using ComponentArrays, Optimization, ForwardDiff
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

The frontend sums the system's `costs` and replaces each definite integral with a
[quadrature state](@ref shooting_quadratures). Equations in `constraints` become
equality constraints, and inequalities become residuals bounded above by zero.
Timed reads such as `x(2.0)` follow the same rules as
[timed expressions](@ref timed_expressions), and a bare parameter refers to its
terminal value.

The [ModelingToolkit Lotka–Volterra tutorial](@ref mtk) shows a complete single-
and multiple-shooting example, and the [LTI tutorial](@ref mtk_lti) jointly
optimizes an input and three constant coefficients.

## Controls and parameters

Controls are parameters with a schedule. Write the grid as interval **start**
times beginning at `tspan[1]`, like `0.0:0.25:1.75` above, or list only the
interior switching times. Bounds come from the symbol metadata.

Parameters marked `tunable = false` stay fixed, unless you schedule them as
controls. All other parameters are optimized.

## Initial values and time span

A state declared as `x(..) = 0.5 [tunable = true, bounds = (0.3, 1.0)]` gets a free
initial value, starting at `0.5` and bounded by `0.3` and `1.0`. States without
`tunable = true` start from their numerical initial values.

The time span is inferred from the integral bounds, timed reads, and grids; pass
`tspan = (start, stop)` to set it explicitly. Solver options go through
`problem_kwargs`, for example `problem_kwargs = (; abstol = 1e-8, reltol = 1e-8)`.
The frontend does not build MTK's initialization system, so give numerical values
or defaults for all states and parameters.

## Multiple shooting

Add `shooting = [0.5, 1.0, 1.5]` to lift every state, including quadratures, at
those times. The continuity equalities are added automatically, and `ensemblealg`
selects [parallel execution](@ref parallel_shooting). Controls and parameters stay
shared between segments.

!!! note "State bounds and `tunable = false`"
    `tunable = false` only fixes the value at `tspan[1]`. Interior shooting values
    are still created, and a state's `bounds` apply to them.

## Inspecting annotations

`Annotations(sys)` reads the metadata of a system and returns
`(; variables, parameters)`: one `Tunable` or `Constant` annotation per entry, with
its index, symbol, default, bounds, and integrality. Use `Annotations(sys; u0, p)` to
take the values from numerical vectors instead of the symbolic defaults. The
frontend uses these annotations before adding control and shooting schedules.

## Array parameters

The system is compiled with `split = false`, so all parameters are flat scalars.
Array parameters are expanded in column-major order, keeping their defaults, bounds,
and tunability, and names such as `a[1]` keep working in queries, costs, and
constraints. Timed reads need scalar signals, and signals eliminated during
simplification can't be referenced. The `evaluator` keyword selects the
expression compiler, as in the [numerical interface](@ref timed_expressions).
