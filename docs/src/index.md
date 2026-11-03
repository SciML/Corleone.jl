```@raw html
---
layout: home

hero:
  name: Corleone.jl
  text: Differentiable dynamic optimization
  tagline: Build optimal control and experimental design problems with SciML solvers and Lux models.
  image:
    src: /logo.png
    alt: Corleone.jl lion logo
  actions:
    - theme: brand
      text: Get started
      link: /getting_started/
    - theme: alt
      text: Explore tutorials
      link: /tutorials/
    - theme: alt
      text: View on GitHub
      link: https://github.com/SciML/Corleone.jl

features:
  - icon: 🎯
    title: Optimal control
    details: Define control schedules, objectives, and constraints using single or multiple shooting.
    link: /shooting/
  - icon: 🔬
    title: Experimental design
    details: Optimize measurements and combine experiments to learn model parameters with CorleoneOED.
    link: /oed/
  - icon: 🧩
    title: Symbolic models
    details: Build dynamics and integral costs with ModelingToolkit, including parameter bounds and initial values.
    link: /modelingtoolkit/
  - icon: ∇
    title: Automatic differentiation
    details: Choose a gradient backend for your solver and optimization problem using tested configurations.
    link: /automatic_differentiation/
  - icon: ⚡
    title: Parallel shooting
    details: Evaluate independent shooting segments concurrently while retaining continuity constraints.
    link: /parallel_shooting/
  - icon: 📊
    title: Benchmark problems
    details: Prepare, solve, and time optimal control problems with OptimalControlBenchmarks.
    link: /benchmarks/
---
```

## From a model to an optimization problem

Corleone turns differential equation solves into differentiable shooting layers.
Choose parameter schedules, state resets, an objective, and constraints; the layer
supplies trajectories and continuity residuals. SciML solvers handle the dynamics,
and Lux containers hold parameters and state.

Start with a [first shooting layer](getting_started.md), then work through the
[optimal control and experimental design tutorials](tutorials.md).
The [API reference](api.md) covers constructors, fields, and optimizer interfaces.
