# [Reduction, AD, and numerical scope](@id oed_numerics)

## [SVD parameter directions](@id oed_svd)

`svd=true` constructs a fixed direction basis before building the augmented layer.
This is a nominal reduction, not optimization of the basis alongside the design.
The reference computation uses the original, unaugmented problem at `problem.p`
and the supplied algorithm. It differentiates a nominal solve with ForwardDiff,
selects `params`, and includes both state and direct observation sensitivities.

At nominal saved times ``t_i``, it stacks all continuous and discrete observation
rows without sampling weights and forms the right-endpoint time-sum surrogate

```math
F_{\mathrm{ref}}=\sum_{i=1}^{N-1}(t_{i+1}-t_i)
J_{\mathrm{ref}}(t_{i+1})^\top J_{\mathrm{ref}}(t_{i+1}).
```

In particular, discrete observations enter this *reference* as a time-integrated
surrogate, not only at their event grids. It is neither the final mixed-design
information matrix nor a guarantee of its identifiable directions. It does not
use current Lux design parameters, measurement weights, or `st.F_init`.

For ``F_{\mathrm{ref}}=U\operatorname{diag}(\sigma)V^\top``:

1. If `ns` is omitted, choose the first ``n_s`` for which
   ``\sum_{i=1}^{n_s}\sigma_i^2/\sum_i\sigma_i^2\geq\tau_s``, where
   `threshold_singular_values=0.95` by default. A zero reference matrix is rejected
   in this automatic-selection case.
2. An explicit `ns` must be an integer between one and `length(params)`.
3. Start with the first ``n_s`` left singular vectors. Retain a physical-parameter
   row only if at least one entry has absolute value **greater than**
   `threshold_singular_vectors=0.1`; set all other rows to zero.

Both thresholds must lie in ``[0,1)``. The resulting `design.basis` is an
``n_\theta\times n_s`` matrix. The row truncation is not followed by
orthonormalization and does not guarantee full column rank. The selection and
basis are fixed construction metadata; no smooth derivative through changing
singular-vector selection is claimed.

Sensitivities and single-experiment information use these reduced directions:
``G\in\mathbb R^{n_x\times n_s}`` and ``F\in\mathbb R^{n_s\times n_s}``.
A single-experiment prior must use the same reduced coordinates. For a fixed
design with the stated initial-sensitivity assumptions, directional information
corresponds to ``B^\top F_{\mathrm{physical}}B`` (up to numerical error), using
the actual thresholded basis. This is not the full physical information matrix.

!!! note "Reduced information stays an approximation"
    Multi-experiment aggregation uses ``E_eF_eE_e^\top`` in common physical axes.
    This does not recover discarded information or undo row truncation. Reduced
    rank can still leave a singular global matrix; inverse criteria need adequate
    combined directions or a scientifically justified prior. The implementation
    does not construct a pseudoinverse-based covariance criterion automatically.

See the [scientific references](@ref oed_references) for related reduction methods;
the steps above specify this implementation rather than all variants in that literature.

## Constant mass matrices and residual DAEs

The current interface expects vector states and vector physical parameters.
The RHS and observation functions must be compatible with Symbolics tracing.
It supports in-place and out-of-place functions; arbitrary numerical callbacks
or state-dependent branches are not automatically made symbolically traceable.

For a constant mass matrix,

```math
M\dot x=f(x,\theta,v,t),\qquad M\dot G=f_xG+f_\theta B.
```

The augmentation uses one copy of ``M`` for the physical state and each sensitivity
column, plus an identity block for continuous-information entries. `UniformScaling`
and constant matrix representations are supported. Derivatives of a mass matrix
with respect to parameters, state, or time are **not** added. A singular constant
mass matrix still requires a solver and initialization suitable for that DAE.

For a residual-form `DAEProblem`, write ``r(\dot x,x,\theta,v,t)=0``. The implemented
sensitivity residual is

```math
r_{\dot x}\dot G+r_xG+r_\theta B=0.
```

Continuous information uses the residual ``\dot F_c-\sum_m w_mJ_m^\top J_m=0``.
When supplied, `differential_vars` repeats the original flags for each sensitivity
column and appends `true` for each integrated-information entry. When omitted,
it remains `nothing`.

The builder starts from the supplied physical `du0`, zero augmented derivatives,
and a pseudoinverse correction using the initial residual Jacobian with respect
to derivatives. This is not a general DAE consistency or initialization algorithm:
it cannot fix arbitrary inconsistent algebraic states. Supply consistent physical
initial data and a suitable DAE solver. No ModelingToolkit initialization system
or arbitrary reset sensitivity map is generated.

Consistency must also hold for the augmented residual. A parameter-dependent
algebraic initial state may require nonzero initial sensitivities, which this
zero-initial-sensitivity construction does not derive. Consistency of the physical
initial state alone does not establish consistency of that augmentation.

## [Differentiating the design solve](@id oed_ad)

Symbolic augmentation builds numerical equations for ``G`` and ``F``. Separately,
an outer AD backend differentiates the design objective through the numerical
augmented solve and parameter reconstruction. Successful symbolic construction or
one forward evaluation does not establish backend compatibility.

The OED Core tests exercise ForwardDiff gradients through mixed measurement
weights, direct observation dependence, shared-grid multi-experiment parameter
containers, and current prior state. They compare selected gradients with finite
differences or analytic formulas. The tutorials use `AutoForwardDiff()` with the
shared SciMLBase bridge and Tsit5. ForwardDiff is also used during nominal SVD
construction. DAE forward regressions do not establish outer DAE-gradient support.

For an initialized `design`, `ps`, and `st` with nonsingular information, an
explicit forward-mode calculation uses the current vector and supplied state:

```julia
using ComponentArrays, ForwardDiff
q0, reconstruct = pack(default_vectorizer(), ps)
objective(q, state) = ACriterion()(first(
    fisher_information(design, nothing, reconstruct(q), state)))
gradient = ForwardDiff.gradient(q -> objective(q, st), q0)
```

This fragment requires the initialized design from a tutorial. It does not
differentiate grid times or the SVD selection. A different supplied state, including
its current prior, is evaluated by `objective(q, state)` rather than captured as
numerical information during preparation.

The root package's maintained ReverseDiff and Zygote configurations are described
in its [AD guide](https://github.com/SciML/Corleone.jl/blob/main/docs/src/automatic_differentiation.md).
Those root tests do not establish all OED/criterion/DAE combinations. Backend
compatibility depends on the solver, sensitivity algorithm, array types, criterion,
user callbacks, and mutation. No blanket reverse-mode or higher-order OED support
is claimed here.

Schedules, sample indices, grid construction, and saved-layout validation are
discrete metadata. Keep them fixed during differentiation and reprepare the
optimization model after schedule/layout changes. Numerical parameters, fixed
values supplied in state, and current priors remain in the computation; avoid
copied numerical snapshots or stale reconstruction closures. The
[ownership guide](@ref oed_ownership) explains why mutating a trajectory's retained
containers can change later information queries.
