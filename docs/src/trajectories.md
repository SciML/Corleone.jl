# [Trajectories](@id trajectory_guide)

Calling a [`ShootingLayer`](@ref) returns `(trajectory, next_st)`. The
[`Trajectory`](@ref) holds the saved state samples and gives access to the
parameter values used in the solve. It stores samples only; it does not interpolate
between them.

## Reading states and parameters

Numerical problems name their states `:u₁`, `:u₂`, … and their parameters `:p₁`,
`:p₂`, …. Problems with a symbolic system, for example from ModelingToolkit, keep
their own names.

| Query | Result |
|---|---|
| `trajectory.t[i]`, `trajectory.u[i]` | The time and state vector of saved sample `i`. |
| `trajectory[state_symbol]` | The state's values at all saved times. |
| `trajectory[parameter_symbol]` | The parameter's values at its own switching times. |
| `parameter_values_at_time(trajectory, trajectory, t)` | The full parameter vector at time `t`. |

The last query comes from SymbolicIndexingInterface. Parameters are piecewise
constant and right-continuous, and their switching times usually differ from the
state sample times, so don't pair the two series by index.

Read `trajectory.t` and `trajectory.u` together rather than predicting the saved
grid from switching times: solver options such as `saveat` add samples, and not
every integration stop is saved. For values at specific times in an objective, use
[timed expressions](@ref timed_expressions).

## Both sides of a reset

At an interior state reset the trajectory keeps two samples with the same time: the
incoming endpoint and the outgoing start. `reset_indices` points to the
**outgoing** samples, so the incoming value sits just before each one.

```@example reset_samples
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random

problem = ODEProblem((x, p, t) -> [p[1]], [0.0], (0.0, 3.0), [1.0])
layer = ShootingLayer(problem, Tsit5(),
    Tunable(1; kind = Val(:variable), default = [0.5, 1.5]) => [1.0, 2.0],
    Constant(1) => nothing)
ps, st = LuxCore.setup(MersenneTwister(42), layer)
trajectory, next_st = layer(nothing, ps, st)
(; times = trajectory.t, outgoing_indices = trajectory.reset_indices)
```

The saved times are `[0, 1, 1, 2, 2, 3]`. At time 1 the incoming value is about 1
and the assigned value is 0.5:

```@example reset_samples
k = first(trajectory.reset_indices)
@assert isapprox(trajectory.u[k - 1][1], 1.0; atol = 1e-8) # hide
@assert trajectory.u[k][1] == 0.5 # hide
round.([trajectory.u[k - 1][1], trajectory.u[k][1]]; digits = 6)
```

[`shooting_constraints`](@ref) returns the matching residuals, with the sign defined
in [multiple shooting](@ref shooting_formulation):

```@example reset_samples
residuals = shooting_constraints(trajectory)
@assert isapprox(residuals, [-0.5, 0.0]; atol = 1e-8) # hide
round.(residuals; digits = 6)
```

The raw field `trajectory.residuals` stores these residuals per segment, for the
tunable states only; prefer `shooting_constraints`, which also includes
inserted-parameter equalities. The [plotting recipe](@ref trajectoryplot) draws each
segment separately, so jumps appear as gaps rather than vertical lines.

## [Shared data and differentiation](@id trajectory_ownership)

A trajectory keeps references to the layer, the parameters, and the returned Lux
state, and its state arrays are shared with the segment solutions. Parameter values
are looked up on demand from those references.

In practice this means:

- Changing `ps` afterwards changes what parameter queries return, but not the saved
  states. Run the layer again to get a consistent trajectory.
- Don't mutate any of this data between a forward call and its pullback.

`symbolic_container(trajectory)` returns the layer's [`ControlSymbolCache`](@ref),
which answers control and quadrature queries; see [symbolic metadata](@ref symbolic_metadata_guide).

Derivatives are supported with respect to numerical values on a fixed time grid, not
with respect to reset or switching times. See
[automatic differentiation](@ref automatic_differentiation) for backends.

## DataFrames and PEtab-style tables

Load `DataFrames` alongside `Corleone` to activate the optional extension. It uses
[PEtab 2](https://petab.readthedocs.io/en/latest/v2/documentation_data_format.html)
measurement and simulation column names, with one row per observable and time.
A single frame can hold many experiments or trajectories.

For a complete fit to published experimental data, see
[Fit published PEtab measurements with DataFrames](examples/inverse_petab_boehm.md).
That tutorial reads the Boehm STAT5 tables, resolves their noise references, fits
six kinetic rates, and exports the fitted observables.

### Import measurements

```julia
using Corleone, DataFrames

measurements = DataFrame(
    observableId = ["x", "x", "x", "x"],
    experimentId = ["control", "control", "treated", "treated"],
    time = [0.0, 1.0, 0.0, 1.0],
    measurement = [1.0, 3.1, 2.0, 4.2],
    noiseParameters = [0.1, 0.2, 0.1, 0.2],
)
data = experiments_from_dataframe(measurements;
    observables = Dict("x" => 1),
    initial_states = Dict("control" => [1.0], "treated" => [2.0]),
)
# data.control and data.treated are Experiment objects.
batch = Experiments(shooting, data) # shooting is your existing ShootingLayer
```

Experiment IDs become named tuple keys in first-appearance order. IDs must be
nonempty strings or symbols. Without `observables`, observable IDs become state
symbols (such as `"u₁"`); mappings can instead provide state indices, state
symbols, scalar callbacks `(u, p, t) -> y`, or Julia observation expressions.
Without `initial_states`, experiments use the problem's initial state. If supplied,
a mapping must cover every ID. `Experiment(df; observables, u0)` imports a frame
containing exactly one experiment. Views of DataFrames are accepted too.

Rows may be unsorted and observables may use different time grids. Repeated
measurements at the same time remain independent residuals, including rows with
`replicateId` metadata. They are stored in separate observations with unique,
sorted times. Residual ordering follows observable first appearance, then
replicate occurrence, then sorted time; it need not match input row order.
The input frame is copied and left unchanged.

`noiseParameters` is optional. This extension interprets each positive numeric
value or numeric string as a known additive standard deviation; blank or missing
entries have unit weight. With no noise column, observations are unweighted.
This interpretation requires the corresponding PEtab noise model to be additive
with these standard deviations. Parameter IDs, lists of noise parameters,
nonempty `observableParameters`, preequilibration, PEtab 1 condition columns, and
infinite measurement times are rejected. Measurements must be finite real values.
Other metadata columns are ignored.

This is a measurement/simulation table adapter. Supply conditions, parameter
annotations, and control schedules through Corleone's Julia API. For example,
replace one named entry with `(observations = data.control, annotations = ...)`
before passing the specifications to `Experiments`. It does not load PEtab model,
condition, observable, or parameter files, or evaluate PEtab formulas.

### Export trajectories

```julia
frame = DataFrame(trajectory; experiment_id = "control")
# Columns: observableId, experimentId, time, simulation, sampleIndex, resetSide

# Named tuple keys supply IDs. Vectors and tuples also work.
frame = trajectories_dataframe((control = trajectory_a, treated = trajectory_b))
frame = DataFrame([trajectory_a, trajectory_b];
    experiment_ids = ["control", "treated"],
    observables = Dict("x" => 1, "total" => ((u, p, t) -> sum(u))),
)
```

The default exports every state using its symbolic name. Output observable
mappings accept symbols, indices, and scalar callbacks. Use callbacks for derived
quantities; Julia expressions are supported on import only. Output is ordered by
trajectory, observable, then saved sample. Each trajectory keeps its own time
grid; no interpolation is performed. Collections must be nonempty and IDs unique.

`sampleIndex` numbers the saved samples within each trajectory. `resetSide` is
`"incoming"` or `"outgoing"` at resets and `"none"` elsewhere. These additional
columns distinguish two state samples at the same time, which standard PEtab
simulation columns alone cannot distinguish. Callback parameters follow
Corleone's right-continuous parameter convention on both reset samples.
For a standard simulation table, select only `observableId`, `experimentId`,
`time`, and `simulation`, and choose the reset side your consumer expects.
A simulation frame renamed from `simulation` to `measurement` can be imported;
reset pairs will then be interpreted as independent measurements.
