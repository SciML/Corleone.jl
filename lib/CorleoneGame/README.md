# CorleoneGame.jl

CorleoneGame.jl is a component of the [Corleone.jl](https://github.com/SciML/Corleone.jl)
monorepo. It computes approximate open-loop Nash equilibria of N-player dynamic
games by damped sequential best responses, where every best response is an
optimal control problem solved with Corleone.jl (single or multiple shooting;
Ipopt, Uno, or BlockSQP2). The directory also contains the benchmark library of
biological dynamic games described in the preprint

> Sager, Plate, Martensen, Ohl, *CorleoneGame: A library of biological dynamic games with a
> companion Julia solver*, arXiv:2610.06307, <http://arxiv.org/abs/2610.06307>.

## Directory layout

| Path | Content |
| --- | --- |
| [`src/`](src) | `CorleoneGame.jl` (game definition and solver), `utils.jl` (CSV output, plots, replotting, synthetic data, logging), and [`CorleoneGame.md`](src/CorleoneGame.md), the description of the interface |
| [`library/`](library) | The 25 cases of the benchmark library, one directory per case with the case file and the saved results in `output/`; see [`library/README.md`](library/README.md) |
| [`examples/`](examples) | Further games that are not part of the library: retired candidate cases and the co-evolution study; see [`examples/README.md`](examples/README.md) |
| [`scripts/`](scripts) | `run_library.jl` (batch solve of library cases) and `plot_library.jl` (redraw plots from saved CSVs) |
| [`test/`](test) | Package tests, started from `test/runtests.jl` |

## First-time setup

CorleoneGame is used from its own project environment. `Project.toml` lists the
dependencies and their compatible versions (Julia 1.11, Corleone 1). As in the
rest of the monorepo, `Manifest.toml` is not tracked; it is created locally on
the first instantiation. After cloning the repository:

```bash
juliaup add 1.11            # once, if Julia 1.11 is not installed
juliaup default 1.11
cd Corleone.jl/lib/CorleoneGame
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

`Pkg.instantiate()` resolves the dependencies, writes `Manifest.toml`, and
installs the packages into the Julia depot; nothing global is modified otherwise. The first precompilation takes
several minutes. Command-line runs additionally need the system `tee` utility on
`PATH` (used for the dual console/log output).

Check the installation by solving the reference case (a few minutes):

```bash
julia --project=. library/tupelo/tupelo.jl
```

To use the solver from another project, add it as a development dependency, e.g.,
`julia --project=path/to/project -e 'using Pkg; Pkg.develop(path="Corleone.jl/lib/CorleoneGame")'`.

## Running cases

All commands below are run from `Corleone.jl/lib/CorleoneGame`. Every case file has
the same entry points:

```bash
julia --project=. library/CASE/CASE.jl single      # single shooting (default)
julia --project=. library/CASE/CASE.jl multiple    # multiple shooting
julia --project=. library/CASE/CASE.jl synthetic   # synthetic observations from the saved trajectory
```

A single-shooting run writes `library/CASE/output/` (multiple shooting:
`output_multiple/`), including `run.log`, which receives the same stdout, stderr,
and warnings as the console. Each run replaces its previous log.

From Julia, load the package and include a case file:

```julia
using CorleoneGame
include("library/tupelo/tupelo.jl")
result = Tupelo.main(parameters=Tupelo.Parameters())          # writes library/tupelo/output
g = Tupelo.game(Tupelo.Parameters(U_A=2.0, U_B=3.5))           # modified hypothesis
result = solve_game(g; options=Tupelo.algorithm_options())     # no files written
```

Each case module exposes `Parameters` (model, objective weights, restrictions),
`algorithm_options(; kwargs...)` (case-specific `Options`), `game(parameters)`,
and `main`. The general interface (`Game`, `Options`, solvers, warm starts,
outputs, display labels) is described in [`src/CorleoneGame.md`](src/CorleoneGame.md).

### Batch runs and plots

`scripts/run_library.jl` solves the library cases in sequence and accepts the same
optional `single`, `multiple`, or `synthetic` argument. By default its loop runs
over the list `all` (all 25 cases); to solve selected cases only, uncomment them in
the list `enabled` and let the loop run over `enabled` instead.
A full run takes several hours and should be started with one Julia process at a
time on machines with little memory.

```bash
julia --project=. scripts/run_library.jl single
```

`scripts/plot_library.jl` redraws `timeseries.png` and `convergence.png` from the
saved `trajectory.csv` and `iterations.csv` of the cases in its `enabled` list,
without solving any game. Run it after cloning to obtain the figures, since the
repository contains no PNG files:

```bash
julia --project=. scripts/plot_library.jl single
```

For a case with custom parameters, pass the original parameters explicitly:

```julia
using CorleoneGame
include("library/tupelo/tupelo.jl")
replot_outputs(Tupelo.game(Tupelo.Parameters()); folder="library/tupelo/output")
```

## Tests

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The tests check the solver on small analytic games (best responses, equilibria,
grids, solvers and fallbacks, warm starts, point costs, duration parameters,
periodic-return constraints), the utilities (CSV output, plots and display labels,
replotting, synthetic observations, logging), and that every library case
constructs a valid game. Individual cases are not solved. A subset is selected
with the environment variable `GROUP` (`Core`, `Utils`, or `Library`), e.g.,
`GROUP=Core julia --project=. -e 'using Pkg; Pkg.test()'`. The tests do not
establish global equilibria or biological calibration.

## Interpretation

Termination requires feasibility for every player and small normalized
unilateral gains under the configured local response searches. This does not
prove global Nash optimality. The library models are synthetic or uncalibrated
unless stated otherwise in the case files and the preprint.
