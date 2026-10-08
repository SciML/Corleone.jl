# Sequential AD compatibility

Run the registered backends with `Pkg.test("CorleoneBase")` in this package's
environment. The registered backends are ForwardDiff, uncompiled ReverseDiff,
Zygote, Mooncake reverse mode, Mooncake forward mode, and a finite-difference
baseline. SciMLSensitivity is loaded, and
ReverseDiff uses
`SciMLSensitivity.SensitivityADPassThrough()` to trace the inner solves while
retaining full solution metadata for transitions and return-code checks. The
default ReverseDiff adjoint rule returns only a tracked state array, not a full
solution. Compiled ReverseDiff tapes are outside this suite's scope.

For out-of-place ODE stages, the driver normalizes state and parameter arrays
with `ArrayInterface.aos_to_soa` before solving. This keeps tracked states and
broadcast derivatives in the same array representation without changing the
model's in-place/out-of-place form.

Zygote uses the `CorleoneBaseZygoteExtension` pullback for `solve!`, with
SciMLSensitivity's default inner-solve rules. The pullback differentiates
stage accumulation through an internal `Zygote.Buffer`, including transitions
and saved endpoints, while preserving the primal iterator and buffer identity.
Discrete stage indices and return-code decisions carry no derivative. The internal buffer is initialized
from the existing solutions and frozen with `copy` only after solving finishes.
Extension-local identity pullbacks normalize SciML state cotangents to
structural solution cotangents with the full solution field set, so Buffer can
accumulate saved-endpoint and transition-metadata gradients. Supplemental
rule tests cover early termination, failed stages, preallocated buffers, zero
cotangents, and direct
iterator metadata gradients. Structural field-access and iterator-constructor
pullbacks avoid double accumulation through Zygote's mutable-object gradient
cache. The initialization pullback filters keyword names without array mutation;
keyword values remain differentiable.

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
