# [Querying symbolic metadata](@id Symbolic-bookkeeping)

[`ControlSymbolCache`](@ref) wraps a SymbolicIndexingInterface index provider with
control, quadrature, and bounds metadata. Controls remain parameters; quadratures
remain states. Their original indices and ordering are preserved.

```@example symbolic_metadata
using Corleone, OrdinaryDiffEqTsit5, SymbolicIndexingInterface

problem = ODEProblem((x, p, t) -> [p[1], x[1]^2],
    [1.0, 0.0], (0.0, 2.0), [0.5])
layer = ShootingLayer(problem, Tsit5(),
    Tunable(1; lowerbound = 0.0, upperbound = 1.0) => [1.0];
    controls = [1], quadratures = [2])
cache = symbolic_container(layer)
(; controls = control_symbols(cache), control_indices = control_indices(cache),
    quadratures = quadrature_symbols(cache), bounds = default_bounds(cache))
```

Subset queries are Corleone methods on the wrapper, not the parent's SII interface.
Use `symbolic_container(layer)` or `symbolic_container(trajectory)` to retrieve it.
The trajectory shares the layer's cache. `symbolic_container(cache)` returns the
parent, preserving its standard indexing and observed-expression behavior.

## Bounds and quadratures

`bounds = Dict(:p₁ => (0.0, 1.0))` supplies annotation defaults to `ShootingLayer`.
Explicit annotation bounds override them; the cache retains effective bounds.
Keys must be state or parameter symbols because a numerical index alone is ambiguous.
`Corleone.get_bounds(cache, symbol)` returns one pair, defaulting to `(-Inf, Inf)`.
Use `get_bounds(layer, ps, st)` for [optimizer bound trees](optimization_contracts.md#Variable-bounds).

Quadrature metadata permits terminal quadratures as a default objective. It adds
neither a differential equation nor path constraints; supply the
[running-cost dynamics](shooting.md#Integral-costs-and-quadrature-states) explicitly
or use [ModelingToolkit](modelingtoolkit.md).

A standalone `ControlSymbolCache(parent; controls, quadratures, bounds)` retains
the parent and bounds dictionary by reference. `default_values(cache)` accesses
the parent's mutable defaults. Changes to shared metadata are visible through the
layer and its trajectories; see [trajectory ownership](trajectories.md#Ownership-and-differentiation).
