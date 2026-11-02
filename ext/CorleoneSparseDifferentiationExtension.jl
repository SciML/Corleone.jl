module CorleoneSparseDifferentiationExtension

import Corleone
import DifferentiationInterface as DI
import SparseMatrixColorings
import SciMLBase
using SparseArrays: nzrange, rowvals, nonzeros
const ADTypes = SciMLBase.ADTypes

function Corleone.sparse_constraint_options(
        prob::Corleone.CorleoneDynamicOptProblem, ad::ADTypes.AbstractADType,
        initial, reconstruct, st
    )
    ad isa ADTypes.AutoSparse && throw(ArgumentError("Use a dense AD backend with sparse_constraints=true; sparsity is applied only to the constraint Jacobian."))
    isempty(prob.lcons) && return (;)
    pattern = Corleone.constraint_sparsity(prob, initial, reconstruct, st)
    backend = ADTypes.AutoSparse(
        ad; sparsity_detector = ADTypes.KnownJacobianSparsityDetector(pattern),
        coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    )
    f = (q, state) -> Corleone.optimization_constraints(prob, reconstruct(q), state)
    preparation = DI.prepare_jacobian(f, backend, initial, DI.Constant(st))
    # Numerical fixed values can change: pass the current state as a DI context.
    jac = (J, q, state) -> begin
        DI.jacobian!(f, J, preparation, backend, q, DI.Constant(state))
        return nothing
    end
    T = typeof(float(zero(eltype(initial))))
    return (; cons_j = jac, cons_jac_prototype = T.(pattern))
end

function Corleone.sparse_hessian_options(
        prob::Corleone.CorleoneDynamicOptProblem, ad::ADTypes.AbstractADType,
        initial, reconstruct, st
    )
    ad isa ADTypes.AutoSparse && throw(ArgumentError("Use a dense AD backend with sparse_hessian=true; sparsity is applied to the prepared Hessian."))
    pattern = Corleone.lagrangian_sparsity(prob, initial, reconstruct, st)
    backend = ADTypes.AutoSparse(
        ad; sparsity_detector = ADTypes.KnownHessianSparsityDetector(pattern),
        coloring_algorithm = SparseMatrixColorings.GreedyColoringAlgorithm()
    )
    f = (q, sigma, multipliers, state) -> Corleone.optimization_lagrangian(prob, reconstruct(q), state, sigma, multipliers)
    sigma = one(eltype(initial))
    multipliers = zeros(eltype(initial), length(prob.lcons))
    preparation = DI.prepare_hessian(f, backend, initial, DI.Constant(sigma), DI.Constant(multipliers), DI.Constant(st))
    prototype = typeof(float(zero(eltype(initial)))).(pattern)
    buffer = copy(prototype)
    # Canonical vector lag_h order: upper triangle in CSC column order.
    positions = [k for j in axes(buffer, 2) for k in nzrange(buffer, j) if rowvals(buffer)[k] <= j]
    function lag_h!(output, q, sigma, multipliers, state)
        if output isa AbstractVector
            length(output) == length(positions) || throw(DimensionMismatch("Lagrangian Hessian value buffer has the wrong length."))
            DI.hessian!(f, buffer, preparation, backend, q, DI.Constant(sigma), DI.Constant(multipliers), DI.Constant(state))
            values = nonzeros(buffer)
            for (i, k) in enumerate(positions)
                output[i] = values[k]
            end
        else
            DI.hessian!(f, output, preparation, backend, q, DI.Constant(sigma), DI.Constant(multipliers), DI.Constant(state))
        end
        return nothing
    end
    hess = (H, q, state) -> lag_h!(H, q, one(eltype(q)), multipliers, state)
    return (; hess, hess_prototype = prototype, lag_h = lag_h!, lag_hess_prototype = prototype)
end

end
