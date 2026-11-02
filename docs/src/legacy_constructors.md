# Deprecated constructors

The old constructor names remain as **functions**, not types. Each emits a Julia
deprecation message and returns native annotations or a `ShootingLayer`:

| Deprecated call | Result | Native replacement |
|---|---|---|
| `ControlParameter(times; ...)` | `Tunable(...) => switching_times` | Parameter annotation and a switching grid |
| `SingleShootingLayer(problem, algorithm; ...)` | `ShootingLayer` | `ShootingLayer` with annotations |
| `MultipleShootingLayer(problem, algorithm, points...; ...)` | `ShootingLayer` | Scheduled variable annotations and `ensemblealg` |
| `MultipleShootingLayer(layer, points...; ...)` | `ShootingLayer` | Extend the layer's variable schedules |

The returned layers use the new Lux parameter/state layout and return
`(trajectory, state)`. Initial-state matching residuals live in
`trajectory.residuals`. These functions do not restore legacy fields or support
using their names as types in dispatch or `isa` checks.

## Controls and single shooting

```julia
control = ControlParameter([0.0, 1.0, 2.0];
    name = :p₁, controls = [0.2, 0.4, 0.6], bounds = (0.0, 1.0))
layer = SingleShootingLayer(problem, algorithm; controls = [1 => control])
```

The equivalent native definition uses only the boundaries between intervals:

```julia
control = Tunable(1; kind = Val(:parameter),
    default = [0.2, 0.4, 0.6], lowerbound = 0.0, upperbound = 1.0) => [1.0, 2.0]
layer = ShootingLayer(problem, algorithm, control; controls = [1])
```

The deprecated shooting constructor selects the parameter using the outer index,
so the old control `name` need not match the underlying symbolic name. A returned
control definition can also be passed directly to `ShootingLayer` when its `name`
is an actual parameter symbol. A problem's final endpoint is excluded from the
translated control switching grid. At a problem start between legacy grid entries,
the control value active at that time is used; before the first grid entry, the
first control value is used.

Legacy control callbacks `(rng, times, bounds) -> values` are adapted to native
time-based defaults and evaluated during setup. Scalar bounds and uniform bound
vectors translate to scalar annotation bounds. Nonuniform per-time bounds are
rejected because the native annotations do not represent them; construct the
optimizer's per-entry bounds explicitly when migrating.

`tunable_ic` selects variables tunable at the problem start. Unselected variables
pass through the problem's initial state. Non-control parameters remain tunable.
`bounds_ic` and `bounds_p` provide annotation bounds; default guesses for tunable
entries are clamped to those bounds. `bounds_p` indexes non-control parameters in
problem parameter order. Extra keywords, including `tspan`, tolerances, and saving
settings, remake the problem before constructing the native layer.

## Multiple shooting

```julia
layer = MultipleShootingLayer(problem, algorithm, [0.0, 1.0, 2.0, 3.0];
    controls = [1 => control], ensemble_alg = EnsembleThreads())
```

Interior shooting points become tunable resets for every state variable. Points
at the problem start or end do not introduce additional interior resets. The
result supports the native serial, threaded, or distributed block execution.
Parameters and controls are shared across blocks. Original tunable reset values
are retained at their original times; newly lifted values default to the bounded
problem initial values.

`quadrature_indices` becomes the cache's quadrature metadata. Quadratures are
lifted like other states and have continuity residuals; use their terminal values
as accumulated costs. Legacy per-stage quadrature summation, parameter/control
lifting, and stage-based optimizer metadata are not reconstructed by these shims.
Fixed reset schedules cannot be combined with tunable shooting resets on the same
variable using this conversion, since a native per-variable layer has one
fixed/tunable status.

Legacy whole-vector `state_initialization` and `parameter_initialization`
callbacks, and whole-layer multiple-shooting `initialization` callbacks, are
rejected with migration guidance. Express guesses through the scalar, vector,
count-based, or time-based [annotation defaults](initialization.md).
