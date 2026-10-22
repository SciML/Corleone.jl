# CorleoneGame: solver and utilities

This note describes the general interface of `CorleoneGame.jl` (game definition,
numerical solver) and `utils.jl` (output, plotting, synthetic data, logging). It is
independent of the case files in `library/` and `examples/`; see the package
[README](../README.md) for installation and the case collections.

CorleoneGame computes approximate **open-loop Nash equilibria** of N-player
dynamic games by damped sequential best responses. Each best response is an
optimal control problem (OCP) of one player against the fixed controls of the
others, transcribed with Corleone.jl (single or multiple shooting) and solved by
Ipopt, Uno, or BlockSQP2.

## Source files

A. **`CorleoneGame.jl`** defines `Game`, `StateConstraint`, `PointCost`,
   `Options`, and `Workspace`; layer construction; OCP solves; damped
   sequential best responses; independent integration; deviation audits; and
   `solve_game`. `solve_game` writes no files or plots, though it logs progress.
B. **`utils.jl`**, included by `CorleoneGame.jl`, contains CSV/report output,
   plotting, replotting of saved output, synthetic observations, and
   command-line logging (`run_game`, `replot_outputs`, `create_synthetic_data`,
   `run_logged`). Loading the module always loads the plotting dependencies.

Exported names: `Game`, `StateConstraint`, `PointCost`, `Options`, `Workspace`,
`solve_game`, `equilibrium`, `best_response`, `audit`, `simulate`,
`initial_controls`, `make_layer`, `setup_layer`, `constraint_norms`,
`periodic_constraints`, `validate_periodic_clock`, `run_game`, `run_logged`,
`replot_outputs`, `create_synthetic_data`, `player_count`, `player_label`, and
`subscript`. Further helpers (for example `CorleoneGame.write_outputs`,
`CorleoneGame.read_plot_outputs`, `CorleoneGame.timeseries`,
`CorleoneGame.with_run_log`, `CorleoneGame.transfer_controls`, and
`CorleoneGame.validate`) are available with the module prefix.

## Defining a game

`Game` is constructed with keyword arguments:

| Field | Meaning |
| --- | --- |
| `name` | Name used in reports and plot titles |
| `dynamics!(dx,x,u,t)` | In-place right-hand side; `u` holds all control components (followed by the parameters, see below) |
| `x0`, `T` | Full initial state (including zero-initialized quadratures) and horizon |
| `objective(xT,player)` | Cost to **minimize**, evaluated at the terminal state; `objective(xT,player,parameters)` if the game has parameters |
| `owners` | `owners[j]` is the player owning control component `j`; players are numbered `1:N`, each owns at least one component |
| `lower`, `upper`, `initial` | Control bounds and initial constant controls, one entry per component |
| `quadratures` | Indices of zero-initialized accumulated states (running costs, budgets); they must not feed back into the dynamics |
| `state_labels`, `control_labels` | Display labels; displayed states come first in state order |
| `constraints` | Vector of `StateConstraint` (player-specific hypotheses) |
| `point_costs` | Vector of `PointCost` (fixed-time costs) |
| `state_lower`, `state_upper` | Invariant bounds for multiple-shooting node states |
| `gain_scales` | One positive scale per player for normalized gains (default ones) |
| `running_objectives` | Quadrature indices written as `phi_X_running` columns |
| `decode`, `validate` | Map numerical to displayed states (quadratures unchanged); check a decoded trajectory |
| `parameters`, `parameter_owners`, `parameter_labels`, `bounds_p`, `time_parameter` | Optional player-owned constant decisions |
| `metadata` | Arbitrary object written to `report.txt`, typically the case parameters |

Controls are matrices `(component, interval)` on an equidistant grid. Players
are labeled A, B, C, … in output and plots; display labels can use Unicode
subscripts, e.g., `"u$(subscript(2))"`, where indices 1, 2, 3 denote players A, B, C.

`StateConstraint(player, index, label, lower=-Inf, upper=Inf, path=false)`
bounds the terminal value of state `index`, or the trajectory sampled at the
transcription points if `path=true`. A constraint belongs to one player and is
imposed only in that player's OCP; a restriction shared by several players is
entered once per player. State-dependent shared constraints therefore lead to a
generalized Nash interpretation.

`periodic_constraints(x0, nplayers; tolerance, log_indices=Int[])` returns two
terminal inequalities per physical state and player, i.e., a return band
`|x_j(T)-x_j(0)| <= tolerance` (in log coordinates for `log_indices`); its labels
start with `!`, so the band is not plotted. Initial values remain fixed, so
closure alone does not imply stability across cycles.
`validate_periodic_clock(T, period, amplitude)` checks that an externally forced
horizon covers an integer number of forcing periods.

### Fixed-time objective features

`PointCost(player=i, time=t, cost=x->...)` adds a cost on states at a fixed
time. The terminal callback contains only terminal terms and accumulated running
costs; `CorleoneGame.trajectory_objective` combines them with point costs
consistently for optimization and independent audits. Point-cost times must be
control-grid nodes on every grid used (otherwise an error is raised); they are
evaluated exactly, without nearest-sample approximation. Point costs must not
depend on quadratures, which preserves stage-local Hessian structure.

### Player-owned constant parameters and free end time

`parameters`, `parameter_owners`, `parameter_labels`, and `bounds_p` define
optional constant decisions. With parameters, the dynamics receive
`[controls; parameters]` and the objective is `objective(xT, player, parameters)`;
with empty parameter vectors the two-argument callback is used. Best responses
expose only the owning player's parameter bounds; other players' parameters are
fixed. Corleone's multiple-shooting constraints match parameter copies between
stages.

`simulate`, `make_layer`, `best_response`, `audit`, `equilibrium`, `solve_game`,
and `run_game` accept `parameters=g.parameters`. Responses, trajectories, runs,
and snapshots retain the optimized vector; audits evaluate each deviation with
its own candidate vector. Refinement transfers the optimized parameters with the
control policy.

For free-time games, set `T=1` and `time_parameter=k` to declare `parameters[k]`
as the duration `D`, so that physical time is `t = D s` for normalized time
`s ∈ [0,1]`. The dynamics and running-cost quadratures must multiply their
physical rates by `D` explicitly; the core performs no hidden time scaling.
Plots and the primary CSV time columns use physical time; free-time CSVs
additionally contain `normalized_time` and constant `parameter_j` columns.

### Block-structured Hessians

For BlockSQP2, terminal objectives must be affine in accumulated quadratures to
preserve stage Hessian blocks. Nonlinear terminal costs in physical states are
supported; nonlinear functions of accumulated quadratures are outside this
block-structured interface.

## Solving

```julia
using CorleoneGame
g = Game(; ...)                         # see above
options = Options(shooting=:multiple, shooting_intervals=10,
    intervals=20, refined_intervals=40, nlp_solver=:blocksqp, warm_start=true)
workspace = Workspace()
result = solve_game(g; options, workspace)        # no files written
# result = run_game(g; options, workspace, folder="output")  # CSVs, report, plots
```

`solve_game` returns `(; game, runs, refined_check, converged)`. `runs` holds one
result for the coarse grid and, if `refined_intervals > intervals`, one for the
refined grid, each with `controls`, `parameters`, `trajectory`, `history`
(objectives, normalized gains, constraint norms per round), `snapshots`
(damped profile per round), the final `check`, and `converged`.
`refined_check` is currently always `nothing`; the extra audit of the
transferred coarse policy is disabled in the source because of its cost.

Lower-level calls:

- `equilibrium(g; options, n, initial, workspace, parameters, on_round)` solves one control grid;
  `on_round(n, history_entry, snapshot)` is called after each round.
- `best_response(g, controls, player; options, workspace, start=nothing, parameters)` solves one OCP;
  `start ∈ [0,1]` replaces the owned controls by a constant fraction of their bounds.
- `audit(g, controls; options, workspace, multistart=false, parameters)` measures unilateral gains.
- `simulate(g, controls; samples=16, parameters)` integrates a fixed policy independently (Tsit5,
  tolerances `1e-10`, restarted on each control interval).
- `initial_controls(g, n)`, `make_layer`, `setup_layer`, and `constraint_norms` expose the transcription.

### Algorithm

In each round, players respond sequentially; the owned controls (and owned
parameters) are updated as `(1-damping)·old + damping·response`. After every round
the damped profile is integrated independently and audited: each player's best
response against it gives the unilateral gain (positive cost decrease), divided
by `gain_scales`. The iteration stops when the profile is feasible for every
player and all normalized gains are at most `tolerance`. At `max_rounds`, the last
profile is returned with `converged=false` and a warning.

On the refined grid, the coarse policy is transferred by averaging over each new
interval. This is exact when the new grid refines the old one and an
approximation otherwise; the refined solve then iterates again.

### Grids

All grids are equidistant. In multiple-shooting mode, `shooting_intervals` is the
number of shooting stages, and every control grid (`intervals`,
`refined_intervals`, and the actual control matrix) must be an integer multiple of
it. Each stage may contain several control intervals and control components;
shooting node states are initialized from an independent simulation at the stage
boundaries. No divisibility between the coarse and refined control grids is
required.

### Options

| Option | Default | Meaning |
| --- | --- | --- |
| `shooting` | `:single` | `:single` or `:multiple` |
| `intervals`, `refined_intervals` | 10, 40 | Coarse and refined control grids; `refined_intervals=intervals` disables the second solve |
| `shooting_intervals` | 10 | Number of multiple-shooting stages |
| `max_rounds`, `damping` | 200, 0.5 | Round limit and damping of sequential best responses |
| `tolerance` | `2e-5` | Threshold for normalized unilateral gains |
| `feasibility_tolerance` | `1e-6` | Independent constraint and shooting-continuity tolerance |
| `max_iters` | 300 | NLP iterations, unless overridden by solver-specific attributes |
| `ode_tolerance` | `1e-9` | Integrator tolerance inside the transcription |
| `constraint_samples`, `audit_samples` | 4, 8 | Samples per control interval for path constraints and audits |
| `nlp_solver` | `:auto` | Ipopt for single, BlockSQP2 for multiple shooting; or `:ipopt`, `:uno`, `:blocksqp` |
| `fallback_solver` | `:ipopt` | Retry after an unsuccessful return code; `nothing` disables it |
| `solver_options`, `fallback_options` | `(;)` | Named tuples overriding the solver defaults |
| `extra_starts` | `[0.0,0.5,1.0]` | Extra constant starts (fractions of each owned component's range) for multistart audits |
| `audit_every_round` | `false` | `true` uses `extra_starts` in every round's audit |
| `warm_start` | `false` | Reuse compatible per-player primal OCP solutions from the `Workspace` |
| `use_synthetic` | `false` | Selects the observation file overlaid in `forward.png` (see Outputs) |

With the current source, `extra_starts` are used only if `audit_every_round=true`
or in an explicit `audit(...; multistart=true)`; the multistart audit at
tentative convergence is disabled.

### Solvers

Ipopt and Uno support both transcriptions; BlockSQP2 requires multiple shooting
and receives Corleone's stage Hessian blocks and variable-block metadata
(condensing remains disabled). Ipopt uses limited-memory Hessians by default
(`solver_options=(;hessian_approximation="exact")` selects exact AD Hessians).
Uno uses `preset="filtersqp"` via
[its Julia interface](https://github.com/cvanaret/Uno/blob/main/interfaces/Julia/README.md).
A custom `BlockSQP2.sparse_options()` object can be passed as
`solver_options=(;options=my_options)`.

The fallback solver starts from the failed solver's finite primal iterate,
projected to the current bounds. If the final return code is still unsuccessful,
a message is printed and the solution is accepted only if the subsequent
independent checks pass: feasibility of the responding player's constraints,
shooting continuity, and agreement of transcription and independently
integrated objectives. A failed check raises an error. Each response prints one
line with player, start, warm-start flag, total NLP iterations, process CPU time
of the solver attempts, CPU time of the setup (including compilation), and the
final status; the returned named tuple also lists all solver `attempts`.

### Repeated solves and warm starts

Game construction is separate from options and workspace state. For repeated
solves, e.g., in inverse dynamic games, construct a new `Game` with updated
weights or constraint hypotheses and reuse a `Workspace`; all closures, bounds,
and differentiation functions are rebuilt from the new game:

```julia
next_result = solve_game(updated_game; options, initial=result.runs[1].controls, workspace)
```

With `warm_start=true`, the workspace stores each player's last independently
validated OCP solution, and a compatible solve reuses its owned controls and
shooting states; opponent controls come from the current policy, and changed
bounds are respected. Changes in grids, state dimension, ownership, quadrature
layout, horizon, or parameter ownership invalidate reuse automatically; explicit
`start` values replace the owned controls. Use `empty!(workspace)` to reset it,
and a separate workspace per concurrent solve (it is not thread-safe). Warm
starts are primal only: dual multipliers, quasi-Newton matrices, and
factorizations are not retained.

## Interpretation

Positive gains are unilateral decreases of minimized cost found by the configured
local response searches. Termination therefore does not prove global Nash
optimality. Damping can destroy feasibility, so every round is audited
independently. Path constraints are sampled during optimization and checked by
denser, tighter independent integration, not continuously in time.

## Outputs (`utils.jl`)

`run_game(g; options, folder, initial, workspace, parameters)` calls `solve_game`
and writes into `folder`:

- `iterations.csv` (grid, round, objectives, normalized gains, constraint norms;
  appended during the run) and `iterations/controls_n<N>_iter<k>.csv` (damped
  profile of each round), so an interrupted run leaves a partial record;
- `trajectory.csv` (physical time, displayed states, running objectives, controls,
  numerical states, and, for parameterized games, `normalized_time` and
  `parameter_j`), `controls.csv`, and `report.txt`;
- `timeseries.png` (states, controls, constraint slacks), `convergence.png`,
  and `iterations/timeseries_n<N>_iter<k>.png` with common axis limits;
- `forward.png` if an observation file exists in `folder`
  (`synthetic.csv` with `use_synthetic=true`, otherwise `<lowercase name>_synthetic.csv`).

`replot_outputs(g; folder)` reads `trajectory.csv` and `iterations.csv` and
replaces only `timeseries.png` and `convergence.png` with the current labels and
layout, without optimization. CSVs without numerical-state columns are
reconstructed by one forward simulation of the saved controls, which is checked
against the saved states; this requires the original model parameters.
Incompatible data raise an error.

`create_synthetic_data(g, folder; sigma=0.02, seed=20260909, measurement_times=nothing,
output="synthetic.csv")` adds Gaussian noise (reproducible seed) to the displayed
states of the most recent `trajectory.csv` in `folder`, `../output`, or
`../output_multiple`, linearly interpolated at the measurement times
(default: 20 equidistant times).

### Display labels

State labels, control labels, and `StateConstraint.label` accept two prefixes:

- `! plant biomass`: omit the curve and its legend entry;
- `SCALE 10 plant biomass`: multiply displayed values by 10 and show `plant biomass`.

They affect plots only; constraint activity is evaluated from the unscaled slack.
Each panel shows the first 16 visible entries; legends use two columns above
eight entries. Identical shared constraint bounds are shown once with all owners,
equalities as one signed residual. The constraint panel is omitted if no visible
constraint remains.

### Command-line runs and logging

`run_logged(main, directory; args=ARGS)` implements the common case entry point
`case.jl [single|multiple|synthetic]`: it calls `main(; shooting, folder)` or
`main(; synthetic=true, folder)` with `folder = directory/output` (or
`output_multiple`) and copies Julia and native-library stdout/stderr, warnings,
and the run summary to the console and `folder/run.log`, replacing any previous
log. `CorleoneGame.with_run_log(f, logpath)` provides the same duplication for
other functions. Both require the system `tee` utility on `PATH`.
