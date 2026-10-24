# Symbolic bookkeeping

`ControlSymbolCache` wraps a SymbolicIndexingInterface index provider with control,
Lagrange-state, and bounds metadata. Controls are a subset of parameters; Lagrange
variables are a subset of states representing running-cost integrals. Both keep
their original kind, index, and ordering. The numerical state and parameter
layouts are those of the underlying problem.

```julia
using Corleone, OrdinaryDiffEqTsit5, SymbolicIndexingInterface

prob = ODEProblem((u, p, t) -> [p[1], u[1]^2],
    [1.0, 0.0], (0.0, 2.0), [0.5])
layer = ShootingLayer(prob, Tsit5(),
    Tunable(1; kind = Val(:parameter), lowerbound = 0.0, upperbound = 1.0) => [1.0];
    controls = [1], lagrange_variables = [2])

sys = symbolic_container(layer)
control_symbols(sys)       # [:p₁]
control_indices(sys)       # [1], the original parameter index
lagrange_symbols(sys)      # [:u₂]
lagrange_indices(sys)      # [2], the original state index
default_bounds(sys)       # Dict(:p₁ => (0.0, 1.0))
is_parameter(sys, :p₁)     # true
variable_symbols(sys)     # the original state symbols
```

The subsets accept symbols or original indices and are independent of whether a
definition is tunable, fixed, or time dependent. They are explicit metadata;
continuity constraints, accumulated-cost handling, and optimization bounds remain
part of the user's objective and constraint definitions.

`ShootingLayer` also accepts `bounds = Dict(:symbol => (lower, upper), ...)`.
Annotations inherit missing bounds from this dictionary and override them when
explicit bounds are supplied. The layer retains the effective annotation bounds
in its symbolic cache. Keys must be state or parameter symbols, since a numeric
index alone would not distinguish states from parameters.

You can construct `ControlSymbolCache(container; controls, lagrange_variables,
bounds)` separately, or pass a differential equation problem as its first
argument. Standalone construction retains the parent and bounds dictionary by
reference. `default_values(cache)` accesses the parent's mutable defaults.
`Corleone.get_bounds(cache, symbol)` returns a bound pair, defaulting to
`(-Inf, Inf)` for an unspecified symbol.

SymbolicIndexingInterface forwards index-provider queries through
`symbolic_container(cache)`, which returns the parent. This also preserves
observed expressions, time-series parameter indices, and other parent behavior.
Corleone retains this wrapper when extracting symbolic metadata from a problem.
The trajectory exposes the same wrapper as its layer through
`symbolic_container(trajectory)`. Metadata is resolved during layer construction;
the forward solve uses the existing numerical buffers and arithmetic.

The control and Lagrange-symbol queries are Corleone methods defined only for
`ControlSymbolCache`. They are separate from SII's standard index-provider
interface. Retrieve the wrapper before querying its bookkeeping data:

```julia
using LuxCore, Random
ps, st = LuxCore.setup(MersenneTwister(1), layer)
trajectory, _ = layer(nothing, ps, st)
cache = symbolic_container(trajectory)
control_symbols(cache)        # [:p₁]
control_indices(cache)        # [1]
lagrange_symbols(cache)       # [:u₂]
lagrange_indices(cache)       # [2]
default_bounds(cache)         # the same dictionary as default_bounds(layer.sys)
Corleone.get_bounds(cache, :p₁) # (0.0, 1.0)
```

The trajectory shares the layer's cache, including its subset vectors and bounds
dictionary. Updates to that metadata are visible through either object.

```@docs
ControlSymbolCache
```
