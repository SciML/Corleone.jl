module CorleoneOptimizationBaseExtension

import Corleone
import OptimizationBase

# Delegate ordering to OptimizationBase: vector-form lag_h values must use the
# upper triangle in exactly the same CSC order as the declared prototype.
function OptimizationBase.lag_hess_structure(prob::Corleone.CorleoneDynamicOptProblem; kwargs...)
    prototype = Corleone.lagrangian_hessian_sparsity(prob; kwargs...)
    return OptimizationBase.lag_hess_structure(prototype)
end

end
