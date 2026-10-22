module CorleoneParameterEstimationOptimizationExtension

using CorleoneParameterEstimation
using Corleone
using ComponentArrays
using LuxCore
using Optimization
using SciMLBase
using Random

# Build a scalar least-squares objective `0.5 * sum(residual.^2)` so the
# parameter-estimation layer can also be optimized with any Optimization.jl solver
# (e.g. `BFGS`, `LBFGS`). This complements the primary NonlinearSolve least-squares
# interface and mirrors how CorleoneOED exposes its OptimizationProblem.

function Optimization.OptimizationProblem(
        pe::ParameterEstimationLayer;
        AD::Optimization.ADTypes.AbstractADType = AutoForwardDiff(),
        u0::Union{Nothing, AbstractVector} = nothing,
        kwargs...
    )
    prob = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(pe; kwargs...)
    (; x0, lb, ub, residual_oop) = prob
    x0 = isnothing(u0) ? x0 : u0

    # Use the out-of-place residual so the objective is compatible with reverse-mode AD
    # backends (`AutoZygote`, ...) as well as forward-mode ones (`AutoForwardDiff`).
    objective = let residual_oop = residual_oop
        (x, p) -> 0.5 * sum(abs2, residual_oop(x, p))
    end

    optf = OptimizationFunction(objective, AD)
    # The scalar objective already closes over everything it needs (the residuals),
    # so nothing is passed as the fixed parameters `p` of the OptimizationProblem.
    #
    # Trivial (`±Inf`) bounds are omitted so that unconstrained solvers such as `Adam`
    # (which reject box constraints) can be used. This mirrors the behaviour of the
    # `NonlinearLeastSquaresProblem` constructor, which also drops trivial bounds. Finite
    # bounds are kept, so bounded solvers (e.g. `LBFGSB`, `Ipopt`) still enforce them.
    finite_bounds = any(CorleoneParameterEstimation._has_finite_bounds, (lb, ub))
    if finite_bounds
        return OptimizationProblem(optf, x0, nothing; lb = lb, ub = ub)
    end
    return OptimizationProblem(optf, x0, nothing)
end

end
