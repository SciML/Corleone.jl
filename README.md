# Corleone.jl

[![Latest Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://sciml.github.io/Corleone.jl/dev/)
[![code style: runic](https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black)](https://github.com/fredrikekre/Runic.jl)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

Corleone turns differential-equation solves into differentiable
[Lux](https://github.com/LuxDL/Lux.jl) shooting layers for dynamic optimization.
Declare which parameters are controls, which initial states are free, and where the
horizon is split into shooting segments; Corleone builds the layer and the
optimization model, and [Optimization.jl](https://github.com/SciML/Optimization.jl)
chooses the values.

- **Optimal control** with single, multiple, and multi-stage shooting
- **Parameter estimation** across experiments with
  [CorleoneInverse](lib/CorleoneInverse)
- **Optimal experimental design** with [CorleoneOED](lib/CorleoneOED)
- **Universal differential equations** with Lux neural networks
- **ModelingToolkit** models, sparse derivatives, and threaded or distributed shooting

## Installation

```julia
using Pkg
Pkg.add(["Corleone", "OrdinaryDiffEqTsit5", "LuxCore"])
```

## Example

Choose a piecewise-constant control so that the state of `dx/dt = p(t)` passes through
0.5 at `t = 1` and reaches 1.5 at `t = 2`:

```julia
using Corleone, LuxCore, OrdinaryDiffEqTsit5, Random
using ComponentArrays, ForwardDiff, Optimization, OptimizationMOI, Ipopt

problem = ODEProblem((x, p, t) -> [p[1]], [0.0], (0.0, 2.0), [0.25])
control = Tunable(1; default = [0.25, 0.75], lowerbound = 0.0, upperbound = 1.0) => [1.0]
layer = ShootingLayer(problem, Tsit5(), control; controls = [1])
ps, st = LuxCore.setup(MersenneTwister(42), layer)

model = CorleoneDynamicOptProblem(layer, :((u₁(1.0) - 0.5)^2 + (u₁(2.0) - 1.5)^2); ps, st)
result = solve(OptimizationProblem(model, AutoForwardDiff()), Ipopt.Optimizer())
```

See the [documentation](https://sciml.github.io/Corleone.jl/dev/) for the full
[getting started guide](https://sciml.github.io/Corleone.jl/dev/getting_started/),
tutorials, and the API reference.
