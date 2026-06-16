# Control parameters become Lux layers, composed via a ControlSet container

**Status**: accepted

Today, `ControlParameter` is a plain struct, and `SingleShootingLayer` flattens however many of them are passed in into one anonymous `Vector{Float64}` (`ps.controls`), tracking their positions in `problem.p` via a separately-maintained, position-aligned `control_indices::Vector{Int64}`. Per-segment ODE parameters are reconstructed at solve time inside generated functions, via an `index_grid`/`tspans`/`parameter_vector`-closure trio built in `initialstates`. This loses each control's identity and spreads its wiring (name, index, bounds) across parallel structures that must stay in sync by position.

We decided to make `ControlParameter` itself a `LuxCore.AbstractLuxLayer`, keeping its existing name and constructor shape but adding an `index::Int64` field (the control's position in the problem's tunable parameter vector, supplied by the user at construction — not assigned later by `SingleShootingLayer`). Its `initialparameters` returns the bare control-values vector directly (no field wrapper). Several of these are composed in a new `ControlSet` container layer, keyed by control name; composing them via standard Lux/NamedTuple semantics naturally produces `ps.controls = (; NAME = vals1, NAME_2 = vals2)` and named bounds `(; NAME = (lb1, ub1), ...)`, instead of today's flat, unnamed vectors.

`ControlSet` also takes over the per-segment reconstruction work: its `initialstates` computes segment boundaries once (same merge algorithm as today's `build_index_grid`/`collect_tspans`, just keyed by name instead of position), and its forward pass, given the live `ps`, emits a fresh **segment grid** — `((tspan_1, p_1), ..., (tspan_N, p_N))` — with each `p_i` reconstructed via `SciMLStructures` repacking. `SingleShootingLayer` calls `ControlSet` first and feeds the segment grid into `_sequential_solve`, replacing the old closure-based indirection. `SingleShootingLayer` collapses to a single `controls::ControlSet` field (no more separate `control_indices`).

We also dropped the existing escape hatch for controls whose index falls outside the tunable parameter vector (controls that fed only a loss/constraint, never the RHS, via `active_controls` filtering in `src/single_shooting.jl`). Going forward, every control must drive the RHS; a quantity that should only affect a loss/constraint should be modeled as a plain optimization parameter, not a `ControlParameter`.

## Considered options

- **Keep the flat vector, just relocate the merge logic.** Rejected — doesn't address the actual goal (restoring per-control structure/identity).
- **Defer `index` assignment to `SingleShootingLayer`, keep `ControlParameter` index-less.** Rejected in favor of supplying `index` at `ControlParameter` construction, so wiring information lives in one place and `SingleShootingLayer`'s `controls` keyword takes ready-made layers rather than `(idx, ControlParameter)` pairs.
- **Generic `LuxCore.AbstractLuxContainerLayer` recursion for `SingleShootingLayer` itself.** Rejected for now — `SingleShootingLayer` still needs hand-written `initialparameters`/`initialstates` to compute `u0`/`p`, which aren't sub-layers, so the existing hand-written-merge style was kept and only extended to delegate the `controls` key to `ControlSet`. `ControlSet` itself, by contrast, *does* subtype `LuxCore.AbstractLuxContainerLayer` (the `Lux.Chain`/`Parallel` idiom) since all of its sublayers genuinely are `ControlParameter` layers — it overrides `initialstates` to merge in the extra container-level segment-boundary state and defines its own custom forward call.

## Scope note

This ADR covers `ControlParameter`, `ControlSet`, and `SingleShootingLayer` only. `MultipleShootingLayer`, `CorleoneDynamicOptProblem`, and the `ext/` extensions (`CorleoneComponentArraysExtension`, the MTK extension) still assume the old flat-vector/positional-pairs shape; they need their own follow-up design pass.
