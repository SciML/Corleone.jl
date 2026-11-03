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
    link: /oed/index/
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
    link: /benchmarks/index/
---
```

## Differentiable shooting for dynamic optimization

Corleone turns differential-equation solves into Lux shooting layers for optimal
control and experimental design. Use it to optimize piecewise-constant controls,
model parameters, or shooting states with SciML solvers and trajectory-based
objectives and constraints.

Start with [Getting Started](getting_started.md), follow a [worked tutorial](gallery/index.md),
or use the [Manual](shooting.md) for individual tasks. The [API reference](api.md)
covers constructors and functions. [CorleoneOED](oed/index.md) adds experimental
design; [OptimalControlBenchmarks](benchmarks/index.md) provides test problems.
