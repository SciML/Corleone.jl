module CorleoneSparseDifferentiationExtension

import Corleone
import DifferentiationInterface as DI
import SparseMatrixColorings
import SciMLBase
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

end
