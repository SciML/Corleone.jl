

## Installation

To install `Corleone.jl`, run the following 

```julia
using Pkg
Pkg.add("Corleone")
```

## Optimal Experimental Design

To derive optimal experimental designs [sager_2013_jan_samplingdecisionsoptimum](@cite) simply add `CorleoneOED.jl` by running 

```julia
using Pkg
Pkg.add("CorleoneOED")
```

## Manual sequential workflows

For an independent, lightweight sublibrary without shooting layers, start with
[Sequential problems (CorleoneBase)](@ref corleonebase). The
[manual single-shooting tutorial](@ref base_fishing) explicitly assembles a
bounded Lotka–Volterra fishing optimization from sequential ODE solves.
