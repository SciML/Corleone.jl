# [Constructing and optimizing designs](@id oed_designs)

## Controls, fixed values, and initialization

Construct `OEDLayer(problem, algorithm, definitions...; params, measurements)` or
wrap an existing `ShootingLayer`. Pass control annotations as in Corleone's
[shooting guide](https://github.com/SciML/Corleone.jl/blob/main/docs/src/shooting.md);
`params` selects estimated physical parameters, and `measurements` supplies
observation maps and grids. The [Lotka tutorial](@ref lotka_oed) is a complete example.
Measurement weights are marked as controls automatically.

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
The mathematical sensitivity/information matrices are not optimizer buffers:
each scheduled scalar still has a vector of values. Named property access is
required; avoid private grouping paths. In a multi-experiment ComponentArray, use
`ps.experiments.experiment_1` (or the chosen name): iteration yields numerical
entries, not experiment subtrees.

The shared [trajectory ownership rules](https://github.com/SciML/Corleone.jl/blob/main/docs/src/trajectories.md)
apply. Information queries combine saved augmented states with **current** weights,
direct observation parameters, and priors. Mutating retained containers can mix
old states and new values. Re-evaluate the layer after changes, and avoid mutation
between an AD forward evaluation and its pullback.

## [Optimization and vectorization](@id oed_optimization)

Both single and multiple experiments prepare a shared `CorleoneDynamicOptProblem`
and use the shared SciMLBase optimization bridge. There is no OED-specific
Optimization or ComponentArrays extension. CorleoneOED itself can execute a design,
compute information, and prepare callbacks without importing either optional package.

The [continuous-design tutorial](@ref lotka_oed) shows preparation with
`CorleoneDynamicOptProblem(design, ACriterion(); ps, st, M)`, optimization, and
trajectory reconstruction. Supply one `M` budget per measurement in the
[documented order](@ref oed_measurements).

The shared [optimization guide](https://github.com/SciML/Corleone.jl/blob/main/docs/src/optimization_contracts.md)
defines packing, current-vector reconstruction, bounds, and preparation ownership.
Supply `ps` and `st` together to retain existing guesses. Each objective/constraint
evaluation predicts once; information and budget extraction add no extra solve.

### Constraints and current state

Pass additional trajectory constraints after the criterion, with `lcons` and
`ucons` defining their limits. For example, a terminal population callback
`trajectory -> last(trajectory.u)[1]` can be bounded by `lcons=0.2, ucons=2.0`.

Constraint order is user callbacks, sampling budgets, then automatic shooting
equalities. Single-experiment expressions use the underlying augmented layer;
multi-experiment models use callbacks over named trajectories. Callback signatures,
structured bounds, and expression syntax follow the shared
[constraint guide](https://github.com/SciML/Corleone.jl/blob/main/docs/src/optimization_contracts.md).

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
