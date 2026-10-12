module CorleoneBaseMooncakeExtension

import CorleoneBase
import CommonSolve
import SciMLBase
import Mooncake
using Mooncake: CoDual, primal, tangent, zero_fcodual, NoRData

# Loaded explicitly by the AD tests: adding a package weakdep/extension entry is
# deliberately outside this change's scope. Only deterministic, mode-independent,
# default-output/default-reduction ensembles with read-only hooks are supported.
function serial_trajectories(problem, algorithm, options, forward)
    solver_options = Base.structdiff(options, NamedTuple{(:trajectories, :safetycopy)})
    template = CorleoneBase.get_problem(problem)
    outputs = map(1:options.trajectories) do i
        ctx = trajectory_context(i)
        trajectory = CorleoneBase.prob_func(problem, template, ctx)
        CommonSolve.solve(trajectory, algorithm; solver_options...)
    end
    return SciMLBase.EnsembleSolution(outputs, forward.elapsedTime, forward.converged, forward.stats)
end

# Context creation carries no numerical derivative. The actual hook and its
# captured inputs remain differentiated. Random/seeded ensembles are rejected.
trajectory_context(i) = SciMLBase.EnsembleContext(i, 1, 0, nothing,
    SciMLBase.default_rng_func(SciMLBase.EnsembleContext(i, 1, 0, nothing, nothing, nothing)), nothing)
Mooncake.@zero_adjoint Mooncake.MinimalCtx Tuple{typeof(trajectory_context), Int}

Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{
    typeof(Core.kwcall), NamedTuple, typeof(CommonSolve.solve),
    CorleoneBase.AbstractParallelProblem, Any, SciMLBase.BasicEnsembleAlgorithm,
}

function Mooncake.rrule!!(
        kwcall::CoDual{typeof(Core.kwcall)}, options::CoDual{<:NamedTuple},
        solve::CoDual{typeof(CommonSolve.solve)},
        problem::CoDual{<:CorleoneBase.AbstractParallelProblem},
        algorithm::CoDual, mode::CoDual{<:SciMLBase.BasicEnsembleAlgorithm})
    nt = primal(options)
    for key in (:output_func, :reduction, :u_init, :seed, :rng, :rng_func,
            :batch_size, :pmap_batch_size)
        haskey(nt, key) && throw(ArgumentError("Mooncake ensemble recomputation does not support $key"))
    end
    get(nt, :safetycopy, true) && throw(ArgumentError("Mooncake ensemble recomputation requires safetycopy=false"))
    forward = CommonSolve.solve(primal(problem), primal(algorithm), primal(mode); nt...)
    # Differentiate the actual problem and keyword CoDuals, not copies of their
    # primals: this connects template fields and captured hook/transition inputs
    # back to the caller. The scheduling forward never serializes tangent data.
    rule = Mooncake.build_rrule(serial_trajectories, primal(problem), primal(algorithm), nt, forward)
    result, back = rule(zero_fcodual(serial_trajectories), problem, algorithm, options, zero_fcodual(forward))
    function ensemble_pullback!!(delta)
        _, dproblem, dalgorithm, doptions, _ = back(delta)
        return NoRData(), doptions, NoRData(), dproblem, dalgorithm, NoRData()
    end
    # Downstream structural cotangents seed the recomputation's output fdata;
    # return the genuine requested-mode primal rather than the serial replay.
    return CoDual(forward, tangent(result)), ensemble_pullback!!
end

end
