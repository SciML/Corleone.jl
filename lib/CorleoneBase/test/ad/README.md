# Sequential AD compatibility

Run the registered backends with `Pkg.test("CorleoneBase")` in this package's
environment. The initial registrations are ForwardDiff and a finite-difference
baseline; this does not claim reverse-mode compatibility of the mutable driver.

The reusable suite accepts any DifferentiationInterface backend:

Each backend runs all four combinations: linear and nonlinear models, each
with out-of-place `f(u, p, t)` and in-place `f!(du, u, p, t)` dynamics. They share
the same loss, analytic references, and primal/prepared/unprepared checks. The
primal checks also verify the problem's in-place flag after every transition.

```julia
using DifferentiationInterface, ForwardDiff, Zygote, SciMLSensitivity
include("suite.jl") # use this file's directory, or an absolute path
using .SequentialADTests

test_sequential_ad(AutoZygote(); solve_kwargs = (; sensealg = ForwardDiffSensitivity()))
```

Load/install the chosen backend and sensitivity packages in the calling test
environment. Available backends with incompatible sequential solves produce test
errors; they are not silently skipped. Supply backend-specific tolerances only
when justified by its numerical differentiation method.

Both cases differentiate `x = [u₁, p₁, p₂, u₃]`. Stage one applies `u₁` and `p₁`
via solve-time remake keywords. Stage two carries the previous endpoint and
changes to `p₂`. Stage three resets to `u₃` and retains `p₂`. Time spans are
`(0, 1)`, `(1, 2)`, and `(2, 3)`, with a fixed three-stage terminal.

The weighted squared-error loss uses every stage endpoint. Closed-form endpoints
and hand-derived Jacobians provide independent gradient references for both
`du/dt = -p*u` and `du/dt = -p*u²`. Prepared evaluations alternate inputs and
repeat the first input to expose stale tapes or leaked mutable state.
