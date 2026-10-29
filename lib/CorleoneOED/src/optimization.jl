"""
$(SIGNATURES)

Prepare an OED objective with the root optimization facade. User constraints follow
Corleone's callback/expression and structured-bound conventions. Optional `M` adds
one upper budget per measurement, continuous then discrete, before the automatic
shooting equalities. Supply `ps` and `st` together to retain their current values.
For a single experiment the prepared model stores its augmented `ShootingLayer`;
for multiple experiments it stores the container. The objective
computes information from the same trajectory without an additional solve.
"""
function Corleone.CorleoneDynamicOptProblem(
        oed::Union{OEDLayer, MultiExperimentLayer}, criterion::AbstractCriterion = ACriterion(), constraints...;
        ps = nothing, st = nothing, rng = Random.default_rng(),
        M = nothing, lcons = nothing, ucons = nothing, kwargs...
    )
    isnothing(ps) == isnothing(st) || throw(ArgumentError("Supply ps and st together."))
    if isnothing(ps)
        ps, st = LuxCore.setup(rng, oed)
    end
    if !isnothing(M)
        length(M) == n_observed(oed) || throw(DimensionMismatch("Supply one sampling budget per measurement."))
        all(x -> x >= 0, M) || throw(ArgumentError("Sampling budgets must be nonnegative."))
        if !isempty(M)
            n = length(constraints)
            if n == 0
                isnothing(lcons) && isnothing(ucons) || throw(ArgumentError("Bounds were supplied without user constraints."))
                lcons, ucons = zeros(length(M)), collect(M)
            else
                (isnothing(lcons) || isnothing(ucons)) && throw(ArgumentError("Supply bounds for the user constraints."))
                lo = n == 1 ? (lcons,) : Tuple(lcons)
                hi = n == 1 ? (ucons,) : Tuple(ucons)
                length(lo) == length(hi) == n || throw(DimensionMismatch("Supply one bound subtree per user constraint."))
                lcons, ucons = (lo..., zeros(length(M))), (hi..., collect(M))
            end
            constraints = (constraints..., trajectory -> sampling_sums(oed, trajectory))
        end
    end
    layer = oed isa OEDLayer ? oed.shooting : oed
    objective = if oed isa OEDLayer
        trajectory -> criterion(fisher_information(oed, trajectory))
    else
        (trajectories, state) -> criterion(experiment_information(oed, trajectories) + state.F_init)
    end
    return CorleoneDynamicOptProblem(layer, objective, constraints...; ps, st, lcons, ucons, kwargs...)
end


"""
    OptimizationProblem(layer::Union{OEDLayer,MultiExperimentLayer}, criterion;
        AD=AutoForwardDiff(), vectorize=default_vectorizer(), ps, st, M, kwargs...)

Prepare an OED model and use Corleone's SciMLBase optimization bridge. Optional `M`
adds budgets in experiment order (continuous then discrete); shooting equalities
follow user constraints and budgets. `integer_weights=true` marks measurement
weights only and works with any vectorizer implementing `pack`. It requires an
integer-capable solver. Supply `ps`/`st` together, or omit both to run setup once.
"""
function SciMLBase.OptimizationProblem(
        layer::Union{OEDLayer, MultiExperimentLayer}, criterion::AbstractCriterion;
        AD = AutoForwardDiff(), vectorize = default_vectorizer(),
        ps = nothing, st = nothing, rng = Random.default_rng(), M = nothing,
        integer_weights::Bool = false, constraints = (), lcons = nothing, ucons = nothing,
        kwargs...
    )
    model = CorleoneDynamicOptProblem(layer, criterion, constraints...; ps, st, rng, M, lcons, ucons)
    if integer_weights
        mask = sampling_integrality!(Corleone.parameter_bound_tree(model.ps, false), layer)
        packed, _ = pack(vectorize, mask)
        return SciMLBase.OptimizationProblem(model, AD; vectorize, int = Bool.(packed), kwargs...)
    end
    return SciMLBase.OptimizationProblem(model, AD; vectorize, kwargs...)
end
