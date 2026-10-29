module CorleoneOEDOptimizationExtension

using CorleoneOED
using ComponentArrays
using Optimization
using LuxCore, Random
import CorleoneOED: n_observed

matching(::OEDLayer, trajectory) = shooting_constraints(trajectory)
matching(layer::MultiExperimentLayer, trajectories) = shooting_constraints(layer, trajectories)

function sampling_integrality!(mask, layer::OEDLayer)
    ids = (map(m -> m.id, layer.measurements.discrete)..., map(m -> m.id, layer.measurements.continuous)...)
    for (name, pc) in pairs(layer.shooting.parameter.parameters)
        pc isa Corleone.FixedLayer && continue
        child = getproperty(mask.parameter.parameters, name).parameter
        if pc.parameter isa NamedTuple
            for id in keys(pc.parameter)
                id in ids && (getproperty(child, id).ps .= true)
            end
        elseif name in ids
            child.ps .= true
        end
    end
    return mask
end
function sampling_integrality!(mask, layer::MultiExperimentLayer)
    for name in keys(layer.experiments)
        sampling_integrality!(getproperty(mask.experiments, name), layer.experiments[name])
    end
    return mask
end

function Optimization.OptimizationProblem(
        layer::OEDLayer, criterion::AbstractCriterion;
        AD = AutoForwardDiff(), vectorize = default_vectorizer(),
        ps = nothing, st = nothing, rng = Random.default_rng(), M = nothing,
        integer_weights::Bool = false, constraints = (), lcons = nothing, ucons = nothing,
        kwargs...
    )
    model = CorleoneDynamicOptProblem(layer, criterion, constraints...; ps, st, rng, M, lcons, ucons)
    if integer_weights
        vectorize === Val(:ComponentArrays) || throw(ArgumentError("Integer weights require ComponentArrays vectorization."))
        mask = sampling_integrality!(zero(ComponentArray(model.ps)), layer)
        return OptimizationProblem(model, AD; vectorize, int = Bool.(getdata(mask)), kwargs...)
    end
    return OptimizationProblem(model, AD; vectorize, kwargs...)
end

"""
    OptimizationProblem(layer::Union{OEDLayer,MultiExperimentLayer}, criterion;
        AD=AutoForwardDiff(), ps, st, M, vectorize=default_vectorizer(), kwargs...)

Pack existing Lux parameters and their bounds using Corleone's vectorizer. `M`
sets nonnegative upper sampling budgets, ordered continuous then discrete within
each experiment. Budgets precede shooting equalities. Omit `ps` and `st` together
to run setup once. `integer_weights=true` marks only measurement weights as integer.
Use a solver supporting integer variables when requesting integrality.
"""
function Optimization.OptimizationProblem(
        layer::MultiExperimentLayer, criterion::AbstractCriterion;
        AD = AutoForwardDiff(), vectorize = default_vectorizer(),
        ps = nothing, st = nothing, rng = Random.default_rng(), M = nothing,
        integer_weights::Bool = false, inplace::Bool = true, kwargs...
    )
    isnothing(ps) == isnothing(st) || throw(ArgumentError("Supply ps and st together."))
    if isnothing(ps)
        ps, st = LuxCore.setup(rng, layer)
    end
    q0, reconstruct = pack(vectorize, ps)
    lower, upper = get_bounds(layer, ps, st)
    lb, _ = pack(vectorize, lower)
    ub, _ = pack(vectorize, upper)
    nshooting = get_number_of_shooting_constraints(layer, ps, st)
    budgets = isnothing(M) ? Float64[] : collect(M)
    isnothing(M) || length(budgets) == n_observed(layer) || throw(DimensionMismatch("Supply one sampling budget per measurement."))
    all(x -> x >= 0, budgets) || throw(ArgumentError("Sampling budgets must be nonnegative."))
    objective = (q, state) -> first(criterion(layer, nothing, reconstruct(q), state))
    constraints = (q, state) -> begin
        current = reconstruct(q)
        trajectories, _ = layer(nothing, current, state)
        residuals = matching(layer, trajectories)
        isempty(budgets) ? residuals : vcat(sampling_sums(layer, trajectories), residuals)
    end
    nconstraints = nshooting + length(budgets)
    cons = nconstraints == 0 ? nothing : inplace ?
        ((res, q, state) -> copyto!(res, constraints(q, state))) : constraints
    fun = OptimizationFunction{inplace}(objective, AD; cons)
    # Retain named axes; broadcasting on ComponentArrays preserves its structure.
    mask = integer_weights ? sampling_integrality!(zero(ComponentArray(ps)), layer) : nothing
    integer_weights && vectorize !== Val(:ComponentArrays) && throw(ArgumentError("Integer weights require ComponentArrays vectorization."))
    return OptimizationProblem(
        fun, q0, st; lb, ub,
        lcons = nconstraints == 0 ? nothing : zeros(nconstraints),
        ucons = nconstraints == 0 ? nothing : vcat(budgets, zeros(nshooting)),
        int = isnothing(mask) ? falses(length(q0)) : Bool.(getdata(mask)), kwargs...
    )
end

end
