# [Constructing and optimizing designs](@id oed_designs)

## Controls, fixed values, and initialization

Construct `OEDLayer(problem, algorithm, definitions...; params, measurements)` or
wrap an existing `ShootingLayer`. For example, the fragment below defines a bounded
physical control at original parameter index 1:

```julia
control = Tunable(1; default=0.0, lowerbound=0.0, upperbound=1.0) => [0.5]
design = OEDLayer(problem, Tsit5(), control;
    params=[2, 3], measurements, controls=[1])
ps, st = LuxCore.setup(Xoshiro(1), design)
trajectory, next_st = design(nothing, ps, st)
```

This fragment assumes a model with at least three parameters and a span containing
`0.5`; the [Lotka tutorial](@ref lotka_oed) provides a complete model. `controls=[1]`
marks the physical control for symbolic bookkeeping and plotting. Measurement
weights are marked as controls automatically. Bounds are queried through
`get_bounds(design, ps, st)` using the supplied parameter structure.

Use `Constant` for prescribed values and `Tunable` for design variables. With the
default OED constructor policy, unannotated original physical parameters remain
fixed at `problem.p`. Fixed values live in Lux state, not in the optimizer vector;
“fixed” does not mean detached if that state is supplied as a differentiated input.
Setup returns parameters and state; evaluation returns the updated state separately.

### Shooting states and continuity

Definitions such as `Tunable([1, 2]; kind=Val(:variable), default=[0.5, 0.7]) => [6.0]`
introduce physical shooting states. At each **interior full physical-state reset**,
OED also introduces tunable sensitivity and continuous-information shooting states.
The ordinary shooting equalities enforce continuity of physical states, sensitivities,
and accumulated continuous information at a feasible design. Discrete information
is assembled afterward from the saved states; it is not a separate reset state.

Initial sensitivity and integrated-information states start at zero. The prior
is stored separately in `st.F_init`. A tunable initial physical value at the global
start is a free design variable, not an estimated initial-condition parameter.
Partial physical-state resets do not automatically lift all augmented states; the
full-state construction is the supported way to split the entire augmented system.

!!! note "Initialize the augmented shooting states"
    Arbitrary shooting guesses need not satisfy continuity or yield meaningful
    positive-definite information for an inverse criterion. After setup, call
    `forward_initialization(Xoshiro(1), design.shooting, ps, st)` to fill existing
    tunable shooting buffers from the reference solve before optimizer preparation.
    This mutates `ps` and returns it; it does not replace the supplied state.

Prescribed physical resets after the initial time, including at the final time,
are rejected: the augmentation does not supply their sensitivity jump maps.
See the shared [shooting formulation](https://github.com/SciML/Corleone.jl/blob/main/docs/src/shooting.md)
and [initialization guide](https://github.com/SciML/Corleone.jl/blob/main/docs/src/initialization.md)
for reset schedules, residual signs, and initialization options.

## [Ownership, storage, and histories](@id oed_ownership)

`OEDLayer` wraps an augmented `ShootingLayer` and delegates its parameter layout.
Numerical parameter buffers are **vectors**, including when several variables or
parameters share a grid and runtime lookup container. The matrices ``G``, ``J``,
and ``F`` are mathematical sensitivity/information objects, not matrix-shaped
optimizer storage. Avoid relying on a particular private grouping path to edit a
named variable; shared-grid grouping can change that nesting.

Parameter containers need named `getproperty` access, not a specific `NamedTuple`
type. A ComponentArray preserves named access around flat numerical storage;
iteration over it yields numbers, not experiment subtrees. For a multi-experiment
parameter tree, use `ps.experiments.experiment_1` (or the supplied experiment name),
not positional iteration of `ps.experiments`.

A trajectory retains references to its layer, `ps`, and `st`. Saved augmented
states share numerical arrays with the segment solutions. Parameter histories are
queried on demand from the *current* containers using
`SymbolicIndexingInterface.parameter_values_at_time(trajectory, trajectory, t)`.
There is no copied parameter-history field. This applies to physical controls,
fixed physical parameters, and measurement weights.

Changing `ps` or numerical state after computing a trajectory changes later
parameter queries without recomputing saved states. Information extraction can
then mix old states with new weights/direct observation parameters. Re-evaluate
the layer when both must describe the same prediction, and do not mutate these
values between an AD forward evaluation and its pullback. Rebinding `ps` or `st`
to a new container does not replace the references in an existing trajectory.
Retain or copy optimizer
vectors deliberately when preserving earlier designs. See the shared
[trajectory ownership guide](https://github.com/SciML/Corleone.jl/blob/main/docs/src/trajectories.md).

## [Optimization and vectorization](@id oed_optimization)

Both single and multiple experiments prepare a shared `CorleoneDynamicOptProblem`
and use the shared SciMLBase optimization bridge. There is no OED-specific
Optimization or ComponentArrays extension. CorleoneOED itself can execute a design,
compute information, and prepare callbacks without importing either optional package.

With ComponentArrays loaded, `default_vectorizer()` selects Corleone's packing
extension. A complete solver workflow appears in each tutorial; its essential steps are:

```julia
using ComponentArrays, Optimization, OptimizationMOI, Ipopt
model = CorleoneDynamicOptProblem(design, ACriterion(); ps, st, M=[4.0, 4.0])
optprob = OptimizationProblem(model, AutoForwardDiff())
result = solve(optprob, Ipopt.Optimizer(); hessian_approximation="limited-memory")
@assert CorleoneOED.SciMLBase.successful_retcode(result)
_, reconstruct = pack(default_vectorizer(), ps)
optimal_ps = reconstruct(result.u)
optimal_trajectory, next_st = design(nothing, optimal_ps, st)
```

This fragment assumes two measurement budgets; use one entry of `M` per
measurement in the [documented order](@ref oed_measurements).

`pack(vectorize, parameters)` returns `(flat_values, reconstruct)`. Reconstruct
using the **current optimizer input** in differentiated callbacks; do not close
over initial numerical values. The default ComponentVector packing reuses backing
storage when possible, and reconstruction wraps the current vector without copying
its numerical values. The same vectorizer flattens parameters, bounds, and integer
masks in matching order. Custom vectorizers implement `Corleone.pack`; they do not
require ComponentArrays. See Corleone's
[packing and optimization contracts](https://github.com/SciML/Corleone.jl/blob/main/docs/src/optimization_contracts.md).

Supply `ps` and `st` together to retain current values; omitting both performs setup
once. Preparing a model performs a prediction to validate callback outputs and
saved layout. Numerical trajectories are not cached for subsequent optimizer calls.
Each objective or constraint evaluation predicts once; extracting information and
budgets from that prediction adds no extra solve.

### Constraints and current state

User constraints follow the root facade's scalar/vector callback, in-place callback,
and structured-bound conventions. An OED example with one constraint is:

```julia
terminal_population = trajectory -> last(trajectory.u)[1]
model = CorleoneDynamicOptProblem(design, ACriterion(), terminal_population;
    ps, st, M=[4.0, 4.0], lcons=0.2, ucons=2.0)
```

For multiple callbacks, supply one top-level bound subtree per callback, such as
`lcons=(0.0, [0.0, 0.0])`. In-place callbacks have the form `f!(res, prediction)`;
their return value is ignored. Constraint order is user callbacks, sampling budgets,
then automatic shooting equalities. Parameter-bound order follows the parameter
vectorizer; it is distinct from this constraint order. Single-experiment timed
expressions use the underlying augmented `ShootingLayer`; multi-experiment models
use callbacks over named trajectories. See the shared
[constraint documentation](https://github.com/SciML/Corleone.jl/blob/main/docs/src/optimization_contracts.md)
for expression syntax and prepared callback rules.

For a custom objective passed through the root facade, both `f(prediction)` and
`f(prediction, st)` are supported; the one-argument form wins if both are applicable.
OED's criterion preparation reads the current prior from the supplied state:
from `trajectory.st.F_init` for a single experiment, and from the optimizer's current
state for a multi-experiment aggregate. To change a prior, pass the updated state
to the optimizer/evaluation, rather than relying on a prior captured when preparing
an objective.

Schedules, sample selection, and layout validation are discrete metadata. Reprepare
an optimization model after changing its schedules, saved layout, or constraint
output sizes; derivatives are defined on the fixed prepared layout.

!!! note "Relaxed weights and integer decisions"
    The tutorials solve continuous relaxations with weights in ``[0,1]``.
    `OptimizationProblem(design, criterion; integer_weights=true, ...)` marks only
    measurement weights as integer, using a native tree flattened with the same
    vectorizer. Physical controls and shooting variables are not marked integer.
    Bounds plus integrality make those weights binary, but enforcing this requires
    an integer-capable solver. Ipopt does not enforce an integer mask. For continuous
    densities, even binary weights budget time spent measuring, not a number of events.

## [Multiple experiments and priors](@id oed_experiments)

`MultiExperimentLayer((first_design, second_design))` gives independent parameter
and state trees; a NamedTuple preserves custom experiment names. Repeating one
layer with `MultiExperimentLayer(design, n)` reuses the layer object,
while Lux setup initializes independent numerical parameter/state trees.

Let ``P`` be the sorted union of estimated original parameter indices. The same
index must have the same physical meaning in every model. For experiment ``e``,
let ``S_e\in\mathbb R^{|P|\times n_{\theta,e}}`` embed its selected physical
coordinates and ``B_e\in\mathbb R^{n_{\theta,e}\times n_{s,e}}`` be its basis.
The stored embedding is ``E_e=S_eB_e``. The aggregate is

```math
F_{\mathrm{multi}}=F_{\mathrm{prior}}+\sum_e E_eF_eE_e^\top,
```

where each local ``F_e`` excludes its child prior. With full bases this embeds
ordinary information into common coordinates; with reduced bases it is the
[documented approximation](@ref oed_svd). A global `st.F_init` has size
`(length(multi.params), length(multi.params))` and is included exactly once.

`multi(nothing, ps, st)` returns `(named_trajectories, next_st)`.
`fisher_information(multi, nothing, ps, st)` solves and returns `(F, next_st)`;
there is no public overload taking only the named trajectories and inferring a
global prior from the child trajectories.

`update_fim(design, completed, state)` returns a merged state with the completed
experiments' information added to `F_init`. Each record has `ps` and may have its
own `st`; otherwise it uses the supplied state. Each record is re-evaluated, with
its existing priors excluded. This avoids counting the same prior repeatedly.
It does not mutate the destination prior. An empty collection returns the original
state; omitting the destination state starts from fresh Lux state. Keep the actual
parameter/state records used in completed experiments for reproducible accumulation.
