```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Corleone.jl 
  text: Fast & Performant Direct Shooting in JuliaLang
  tagline: 🚀 Shoot for the stars with the performance of SciML & the flexibility of Lux.
  actions:
    - theme: brand
      text: Tutorials
      link: tutorials
    - theme: alt
      text: View on GitHub
      link: https://github.com/SciML/Corleone.jl
  image:
    src: logo.png
    alt: Corleone.jl

features:
  - icon: 🚀
    title: Fast & Interoperable
    details: Corleone aims to provide an easy-to-use but still flexible framework to solve various optimal control problems.
    link: /introduction

  - icon: 🔋
    title: SciML 
    details: Build upon the performant tools of SciML, e.g. OrdinaryDiffEq.jl and Optimization.jl, it comes with batteries included.
    link: https://sciml.ai/

  - icon: 🤖 
    title: ML Ready 
    details: Corleone is build upon LuxCore.jl. 
    link: https://lux.csail.mit.edu/stable/

  - icon: 🧪
    title: Optimal Experimental Design
    details: Automatically derive optimal experimental designs for your models, leveraging the capabilities of Symbolics.jl
    link: /oed
---
```


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