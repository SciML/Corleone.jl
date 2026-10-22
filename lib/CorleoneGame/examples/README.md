# CorleoneGame examples

Games built with CorleoneGame that are **not** part of the benchmark library in
[`../library`](../library). Each example has its own directory with its Julia
file and is run from `lib/CorleoneGame`, e.g.,

```bash
julia --project=. examples/vascular/vascular.jl
```

Except for co-evolution, no saved results are included; a run writes
`output/` next to the case file.

## Retired library candidates

These models were developed as library candidates but were not included in the
library or the preprint. They use the same entry points as the library cases
(`single`, `multiple`, `synthetic`).

| Directory | Reason for retirement |
| --- | --- |
| [`geneticCode/`](geneticCode) | Abstract pool-tracking proxy for proofreading; decorative states; no primary source for the mechanism. |
| [`chemotaxis/`](chemotaxis) | No motility or receptor adaptation; exogenous attractant; duplicated the structure of proteinFolding. |
| [`vascular/`](vascular) | Resistance had no feedback on flow; Murray's law not used. |
| [`genomeSize/`](genomeSize) | Modeled expression burden rather than genome size; source did not support the mechanism. |
| [`collectiveMigration/`](collectiveMigration) | The group dispersed and the followers did not influence the leader (one outer round). |
| [`algaehu/`](algaehu) | Removed by author choice; to be investigated in a separate study. |

## Co-evolution of a body and an adaptive implant

[`co-evolution/`](co-evolution) is a separate study on the co-adaptation of two
learners: a body and an adaptive implant co-evolve to an open-loop Nash
equilibrium, and an outer design problem chooses the implant's adaptation
objective (mismatch weight μ) such that the equilibrium is close to a healthy
state.

| Path | Content |
| --- | --- |
| `simple.jl` | Two-player model (one state, control, and objective per player) and the design scan over μ (module `CoEvolutionSimple`) |
| `checks/` | `grid20.jl`, a check of selected designs on the 20-interval grid, and its saved console output |
| `output/` | Saved results of the design scan: `design_scan.csv`, per-design controls and trajectories for two initial policies, `run.log`, `simple.png` |
| `report/` | LaTeX report on the results and the literature context (`report.tex`, `coevolution.bib`, `make_table.py` generating the result table from `output/design_scan.csv`, compiled `report.pdf`) |
| `literature.md` | Literature survey (two-learner problems, co-adaptation, human-in-the-loop learning, adaptive and dual control, learning in games) |
| `nextsteps.md` | Open questions and suggested next steps |

```bash
julia --project=. examples/co-evolution/simple.jl
julia --project=. examples/co-evolution/checks/grid20.jl 2.0 0.5 0.25
```
