```@raw html
---
layout: home

hero:
  name: Corleone.jl
  text: Modular shooting for dynamic optimization and scientific machine learning
  tagline: Build dynamic optimization problems with SciML solvers and Lux models.
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
    details: Optimize piecewise-constant controls, free initial states, and stage durations with single or multiple shooting.
    link: /shooting/
  - icon: 📈
    title: Parameter estimation
    details: Fit model parameters to measurements from one or many experiments, with priors, transforms, and uncertainty estimates in CorleoneInverse.
    link: /inverse/
  - icon: 🔬
    title: Experimental design
    details: Choose controls and measurement times that make parameters identifiable with CorleoneOED.
    link: /oed/
  - icon: 🧠
    title: Scientific machine learning
    details: Embed Lux neural networks in the dynamics and train them like any other parameter.
    link: /examples/lotka_volterra_universal_differential_equation/
  - icon: 🧩
    title: Symbolic models
    details: Write dynamics, integral costs, and constraints in ModelingToolkit and let Corleone build the shooting problem.
    link: /modelingtoolkit/
  - icon: ⚡
    title: Fast derivatives
    details: Use ForwardDiff, ReverseDiff, or Zygote, sparse Jacobians and Hessians, and threaded or distributed shooting.
    link: /automatic_differentiation/
---
```

## Differentiable shooting for dynamic optimization

Corleone turns a differential-equation solve into a Lux layer. You declare which
parameters are controls, which initial states are free, and where the horizon is
split into shooting segments; Corleone builds the layer, and Optimization.jl or
NonlinearSolve.jl chooses the values. The same layer serves optimal control,
parameter estimation, optimal experimental design, and universal differential
equations.

New to Corleone? [Getting Started](getting_started.md) builds and optimizes a first
problem in a few lines. The [tutorials](gallery/index.md) work through complete
applications, the Manual explains each building block, starting with
[parameter schedules](scheduling.md), and the [API reference](api.md) lists every
exported function. [CorleoneInverse](inverse/index.md) and
[CorleoneOED](oed/index.md) add parameter estimation and experimental design.
