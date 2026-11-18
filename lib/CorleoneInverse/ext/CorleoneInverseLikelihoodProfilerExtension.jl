module CorleoneInverseLikelihoodProfilerExtension

using CorleoneInverse: ParameterEstimate
using SciMLBase: OptimizationProblem
import LikelihoodProfiler: ProfileLikelihoodProblem

"""
    ProfileLikelihoodProblem(problem::OptimizationProblem, estimate::ParameterEstimate;
        idxs = nothing, include_controls = false, conf_level = estimate.level, kwargs...)

Profile a fitted CorleoneInverse model using its original optimization problem.
The estimate must have been constructed from the same model and packed layout.
Uses the retained internal optimum, including for transformed parameters.
By default, profile every tunable value except controls; controls remain free
nuisance parameters. Set `include_controls = true` to profile all values, or
pass `idxs` explicitly to select packed indices (including controls).
If control metadata is unavailable, supply `idxs` or `include_controls = true`.

`conf_level` defaults to the estimate's confidence level. Remaining keywords,
including `profile_lower`, `profile_upper`, `df`, and `threshold`, are forwarded
to LikelihoodProfiler. Profile bounds and results use internal coordinates.
Supply finite profile bounds when the optimization bounds are infinite.
The objective is used unchanged: construct it on the likelihood scale required
by LikelihoodProfiler and choose a matching `threshold` when needed.
"""
function ProfileLikelihoodProblem(
        problem::OptimizationProblem, estimate::ParameterEstimate;
        idxs = nothing, include_controls::Bool = false, conf_level = estimate.level, kwargs...
    )
    isnothing(estimate.internal_x) && throw(
        ArgumentError(
            "Profiling requires an estimate constructed from a model and fitted vector."
        )
    )
    length(problem.u0) == length(estimate.internal_x) || throw(
        DimensionMismatch(
            "The optimization problem and estimate must have the same packed parameter layout."
        )
    )
    if isnothing(idxs)
        !include_controls && isnothing(estimate.control_indices) && throw(
            ArgumentError(
                "Default control exclusion requires control metadata; pass idxs or include_controls = true."
            )
        )
        idxs = include_controls ? collect(eachindex(estimate.internal_x)) :
            setdiff(eachindex(estimate.internal_x), estimate.control_indices)
        isempty(idxs) && throw(
            ArgumentError(
                "No non-control parameters to profile; pass idxs or include_controls = true."
            )
        )
    end
    return ProfileLikelihoodProblem(problem, copy(estimate.internal_x); idxs, conf_level, kwargs...)
end

end
