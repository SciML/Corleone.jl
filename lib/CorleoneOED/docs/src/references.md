# [References and attribution](@id oed_references)

## Scientific background

- Sebastian Sager, *Sampling Decisions in Optimum Experimental Design in the Light
  of Pontryagin's Maximum Principle*, SIAM Journal on Control and Optimization
  **51**(4), 3181–3207 (2013).
  [doi:10.1137/110835098](https://doi.org/10.1137/110835098).
  Background on dynamic sampling design and information-gain functions.
- Carl Julius Martensen, Christoph Plate, and Sebastian Sager, *DynamicOED.jl:
  A Julia package for solving optimum experimental design problems*, Journal of
  Open Source Software **9**(98), 6605 (2024).
  [doi:10.21105/joss.06605](https://doi.org/10.21105/joss.06605).
  Scientific/software context; the current CorleoneOED API is specified by this manual.
- Christoph Plate, Carl Julius Martensen, and Sebastian Sager, *Optimal Experimental
  Design for Universal Differential Equations*, IEEE Transactions on Automatic
  Control (2025), [doi:10.1109/TAC.2025.3609533](https://doi.org/10.1109/TAC.2025.3609533).
  [Open preprint: arXiv:2408.07143](https://arxiv.org/abs/2408.07143).
  Related parameter-reduction methods. The [SVD guide](@ref oed_svd) states the
  specific surrogate, thresholds, and approximation used here; this package does
  not implement every method in the paper.

The tutorials adapt the mintOC
[Lotka experimental-design benchmark](https://mintoc.de/index.php?title=Lotka_Experimental_Design)
and [compartmental OED benchmark](https://mintoc.de/index.php?title=Compartmental_OED).
They demonstrate continuous relaxations on the stated grids. Their successful
local solver termination is not a certificate of a globally optimal integer design
or exact reproduction of the benchmark's reference solution.

## Software provenance and license

The measurement abstraction, mixed continuous/discrete design, multi-experiment
work, and DAE augmentation were adapted from **Christoph Plate (@chplate)** in
[AlCap23/Corleone.jl](https://github.com/AlCap23/Corleone.jl), using its `main`
branch at [7443580115](https://github.com/AlCap23/Corleone.jl/commit/7443580115fa432e4996a67b45a9028b3d10e7ed).
See also [the measurement abstraction](https://github.com/AlCap23/Corleone.jl/commit/4d300d3337)
and [multi-experiment work](https://github.com/AlCap23/Corleone.jl/commit/70d8b51e3a).

The current rewrite adapts those contributions to Corleone's shared layer,
trajectory, and optimization contracts. Original MIT license and author credits
are retained in
[LICENSE.md](https://github.com/SciML/Corleone.jl/blob/main/lib/CorleoneOED/LICENSE.md)
and the package metadata. Software provenance is separate from the scientific
references above.
