"""
$(SIGNATURES)

Prepare an OED objective with the root optimization facade. User constraints follow
Corleone's callback/expression and structured-bound conventions. Optional `M` adds
one upper budget per measurement, continuous then discrete, before the automatic
shooting equalities. Supply `ps` and `st` together to retain their current values.
The prepared model stores the underlying augmented `ShootingLayer`; its objective
computes information from the same trajectory without an additional solve.
"""
function Corleone.CorleoneDynamicOptProblem(
        oed::OEDLayer, criterion::AbstractCriterion = ACriterion(), constraints...;
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
    objective = trajectory -> criterion(fisher_information(oed, trajectory))
    return CorleoneDynamicOptProblem(oed.shooting, objective, constraints...; ps, st, lcons, ucons, kwargs...)
end
