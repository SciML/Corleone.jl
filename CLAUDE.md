# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Corleone.jl is a Julia package for formulating and solving dynamic/optimal control
problems built on top of the SciML stack (`SciMLBase`, `SciMLStructures`,
`SymbolicIndexingInterface`) and Lux's layer abstraction (`LuxCore.AbstractLuxLayer`).
Controls and tunable initial conditions/parameters are exposed as Lux-style
parameters/states so the resulting problem can be handed to `Optimization.jl`.

The repo also hosts two sublibraries under `lib/`:
- `lib/CorleoneOED` — optimal experimental design built on top of Corleone.
- `lib/OptimalControlBenchmarks` — benchmark problems.

These sublibraries have their own `Project.toml`/`test` and are tested independently
(see Testing below); they are not part of the main package's dependency graph.

## Commands

This is a Julia package (no JS/Python build steps). From the repo root:

```julia
# Instantiate the main package environment
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run the default ("Core") test group — light tests, no heavy deps
julia --project=. -e 'using Pkg; Pkg.test()'
```

Test groups are driven by `SciMLTesting.run_tests` and selected via the `GROUP` env var
(see `test/runtests.jl` and `test/test_groups.toml`):

```bash
GROUP=Core     julia --project=. test/runtests.jl   # core/local_controls.jl, core/multiple_shooting.jl
GROUP=Examples julia --project=. test/runtests.jl   # heavy: MTK + Ipopt/MOI + SciMLSensitivity, isolated env in test/examples
GROUP=QA       julia --project=. test/runtests.jl   # Aqua.jl code-quality checks, isolated env in test/qa
GROUP=All      julia --project=. test/runtests.jl   # alias for Core only
```

`Examples` and `QA` activate their own sub-environment (`test/examples`, `test/qa`)
so their extra dependencies never leak into the main package's light test resolve.

To run a sublibrary's tests locally, pass its directory name as `GROUP` (this routes
through the sublibrary dispatcher in `test/runtests.jl`, which activates
`lib/<name>` and hands off via `CORLEONE_TEST_GROUP`):

```bash
GROUP=CorleoneOED julia --project=. test/runtests.jl          # CorleoneOED's "Core" group
GROUP=CorleoneOED_QA julia --project=. test/runtests.jl       # CorleoneOED's "QA" group
```

In CI, each `lib/<name>` sublibrary is instead tested directly via the
SciML/.github sublibrary-project-tests workflow against its own `Project.toml`,
not through this dispatcher.

Formatting follows `SciMLStyle` (badge in README); there is no separate lint command
beyond what `Aqua.jl` checks in the `QA` group.

## Architecture

### Layer stack (the core abstraction)

Corleone models a control problem as a chain of Lux layers, mirroring how a neural
network is composed of layers with `initialparameters`/`initialstates`/`(layer)(x, ps, st)`:

- **`ControlParameter`** (`src/local_controls.jl`) — a piecewise-constant control
  discretization: a name, a time grid `t`, initial values (vector or
  `(rng, t, bounds) -> u` function), and bounds (tuple or function of the time grid).
  Helper functions here (`build_index_grid`, `collect_tspans`, `collect_local_controls`,
  `collect_local_control_bounds`) merge multiple `ControlParameter`s that may live on
  different time grids into one combined grid used to drive segmented ODE solves.

- **`SingleShootingLayer`** (`src/single_shooting.jl`) — wraps an
  `SciMLBase.AbstractDEProblem` + algorithm. Some entries of `problem.p` are
  "control indices" driven by `ControlParameter`s instead of being constant;
  some entries of `problem.u0` are "tunable" (free initial conditions). Calling
  the layer (`layer(u0, ps, st)`) integrates the ODE/DAE piecewise across the
  combined control time grid (`_sequential_solve`, generated functions specialized
  on the number of segments `N`) and returns a `Trajectory`. `quadrature_indices`
  mark states that are pure quadratures (don't feed back into the RHS) and are
  treated specially during initial-condition handling.

- **`MultipleShootingLayer`** (`src/multiple_shooting.jl`) — wraps a
  `SingleShootingLayer` and lifts the problem onto disjoint `shooting_intervals`,
  turning interior initial conditions into additional degrees of freedom
  (closed via shooting/continuity constraints). Supports parallel integration of
  intervals via an `EnsembleAlgorithm` (`EnsembleSerial`/`EnsembleThreads`/
  `EnsembleDistributed`, dispatched through `mythreadmap` in `src/Corleone.jl`).
  Initialization of the interior shooting nodes is pluggable — see
  `src/node_initialization.jl` (`random_initialization`, `forward_initialization`,
  `linear_initialization`, `constant_initialization`, `hybrid_initialization`,
  `custom_initialization`, with `default_initialization` as the baseline).

- **`Trajectory`** (`src/trajectory.jl`) — the solution type returned by calling a
  shooting layer. Implements `SymbolicIndexingInterface` (`state_values`,
  `parameter_values`, `current_time`, `symbolic_container`) so trajectory values can
  be queried by symbol (`traj[:x]`) the same way SciML solution objects are.

- **`CorleoneDynamicOptProblem`** (`src/dynprob.jl`) — ties a shooting layer plus a
  symbolic loss expression and path/point constraints (evaluated via
  `SymbolicIndexingInterface.getsym` against the layer's `SymbolCache`) into an
  objective + constraint closure. Shooting/continuity constraints from
  `MultipleShootingLayer` are folded in automatically. This is converted to a real
  `SciMLBase.OptimizationProblem`/`OptimizationFunction` via
  `wrap_functions`/`to_vec`, which are intentionally left undefined in core
  Corleone (`function wrap_functions end`) and only implemented by extensions —
  loading `ComponentArrays.jl` or `Lux`/`Functors`-compatible packages supplies the
  vectorization strategy needed to flatten the nested `ps`/`st` NamedTuples into a
  flat vector for the optimizer.

### Extensions (`ext/`)

Package extensions activate when the corresponding weak dependency is loaded:
- `CorleoneComponentArraysExtension` — implements `wrap_functions`/`to_vec` using
  `ComponentArrays.jl` (the typical way to get a flat parameter vector).
- `CorleoneModelingToolkitExtension` (`ext/MTKExtension/`) — bridges
  `ModelingToolkit.jl` systems into `SingleShootingLayer`/`CorleoneDynamicOptProblem`
  (building the `AbstractDEProblem` and symbol cache from an MTK system).
  `optimal_control.jl` and `utils.jl` split construction logic from helpers.
- `CorleoneMakieExtension` — plotting `Trajectory`s with Makie.

### Symbol handling

Symbolic indexing (`SymbolCache`, `retrieve_symbol_cache`) is used throughout to let
users refer to states/parameters/controls by name rather than positional index, both
when no underlying symbolic system exists (auto-generated `x₁, x₂, ...`, `p₁, p₂, ...`
names in `src/single_shooting.jl`) and when one does (e.g. from MTK).

### Generated functions for shooting

`_sequential_solve` in `src/single_shooting.jl` is implemented as `@generated`
functions specialized on the number `N` of time segments, unrolling the sequential
(or, for `MultipleShootingLayer`, parallel-across-intervals) ODE solves at compile
time rather than looping at runtime — this is performance-sensitive code in the hot
path of every objective/constraint evaluation during optimization.
