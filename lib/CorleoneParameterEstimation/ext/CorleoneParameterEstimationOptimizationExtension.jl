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
    prob = CorleoneParameterEstimation.CorleoneParameterEstimationProblem(pe)
    (; ps0, st, x0, lb, ub, residual) = prob
    x0 = isnothing(u0) ? x0 : u0

    objective = let residual = residual
        (x, p) -> begin
            r = similar(x, CorleoneParameterEstimation.n_observations(
                CorleoneParameterEstimation.get_data(pe)))
            residual(r, x, p)
            return 0.5 * sum(abs2, r)
        end
    end

    optf = OptimizationFunction(objective, AD)
    return OptimizationProblem(optf, x0, st; lb = lb, ub = ub)
end

end
