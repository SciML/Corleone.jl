module CorleoneBaseMooncakeExtension

import CorleoneBase
import CommonSolve
import SciMLBase
import Mooncake
using Mooncake: CoDual, Dual, primal, tangent, zero_dual, zero_fcodual, NoRData

# Loaded automatically as the registered `CorleoneBaseMooncakeExtension`
# weakdep extension when CorleoneBase and Mooncake are both loaded. Only
# deterministic, mode-independent, default-output/default-reduction ensembles
# with read-only hooks are supported.
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
trajectory_context(i) = SciMLBase.EnsembleContext(
    i, 1, 0, nothing,
    SciMLBase.default_rng_func(SciMLBase.EnsembleContext(i, 1, 0, nothing, nothing, nothing)), nothing
)
Mooncake.@zero_adjoint Mooncake.MinimalCtx Tuple{typeof(trajectory_context), Int}

Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{
    typeof(Core.kwcall), NamedTuple, typeof(CommonSolve.solve),
    CorleoneBase.AbstractParallelProblem, Any, SciMLBase.BasicEnsembleAlgorithm,
}

function Mooncake.rrule!!(
        kwcall::CoDual{typeof(Core.kwcall)}, options::CoDual{<:NamedTuple},
        solve::CoDual{typeof(CommonSolve.solve)},
        problem::CoDual{<:CorleoneBase.AbstractParallelProblem},
        algorithm::CoDual, mode::CoDual{<:SciMLBase.BasicEnsembleAlgorithm}
    )
    nt = primal(options)
    for key in (
            :output_func, :reduction, :u_init, :seed, :rng, :rng_func,
            :batch_size, :pmap_batch_size,
        )
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
    # The serial replay's output CoDual is returned unchanged, so all downstream
    # derived rules assert against the recomputation's own (self-consistent)
    # types; its primal values equal the requested-mode ensemble's.
    return result, ensemble_pullback!!
end

# Forward-mode support for AutoMooncakeForward. The MinimalCtx @is_primitive above makes
# the same kwcall signature a DefaultCtx primitive too, so Mooncake's forward mode reaches
# it and requires an frule!! taking Duals. Same contract as the reverse-mode rule: run the
# genuine requested-mode primal ensemble (so threaded/distributed execution is real), then
# differentiate the actual deterministic serial replay with the incoming Duals, so the
# template fields and captured hook/transition inputs stay connected to the caller. The
# replay is a real EnsembleSolution of the same values, so it is returned directly.
function Mooncake.frule!!(
        ::Dual{typeof(Core.kwcall)}, options::Dual{<:NamedTuple},
        ::Dual{typeof(CommonSolve.solve)},
        problem::Dual{<:CorleoneBase.AbstractParallelProblem},
        algorithm::Dual, mode::Dual{<:SciMLBase.BasicEnsembleAlgorithm}
    )
    nt = primal(options)
    for key in (
            :output_func, :reduction, :u_init, :seed, :rng, :rng_func,
            :batch_size, :pmap_batch_size,
        )
        haskey(nt, key) && throw(ArgumentError("Mooncake ensemble recomputation does not support $key"))
    end
    get(nt, :safetycopy, true) && throw(ArgumentError("Mooncake ensemble recomputation requires safetycopy=false"))
    forward = CommonSolve.solve(primal(problem), primal(algorithm), primal(mode); nt...)
    rule = Mooncake.build_frule(serial_trajectories, primal(problem), primal(algorithm), nt, forward)
    result = rule(zero_dual(serial_trajectories), problem, algorithm, options, zero_dual(forward))
    return result
end

# --- Keyword-bearing problems inside Mooncake reverse/forward -----------------
#
# Mooncake's derived rules for `solve(::ODEProblem, alg; kwargs...)` misorder
# `prob.kwargs` when the problem's own kwargs are stored in the ODESolution:
# the traced CoDual expects the canonical `@NamedTuple{...}` keyword order while
# the replayed solve rebuilds it in hash-table order, and the reverse pass's
# `typeassert` fires (`TypeError: in typeassert, expected Mooncake.CoDual{
# ODESolution{...}}`). This reproduces with a plain
# `solve(ODEProblem(f, u0, tspan, p; abstol...), Tsit5(); ...)` gradient under
# Mooncake 0.5.63 + SciMLBase 3.57 + DiffEqBase 7.21 without CorleoneBase
# (the kwargs-in-problem clause of SciMLSensitivity.jl issue #1337). Keywords
# supplied at the solve call itself are unaffected because DiffEqBase's merge
# takes the `isempty(prob.kwargs)` shortcut.
#
# The rules below keep the CorleoneBase integration usable at these pinned
# versions by differentiating a replay in which the problem's solver controls
# are moved from the problem onto the solve call. DiffEqBase merges problem
# keywords with solve keywords, so the replay's primal is numerically identical
# to the genuine solve (same tolerances, same saved solutions), and the moved
# keywords are nondifferentiable solver controls. The replay is returned
# unchanged, so its CoDual types stay self-consistent for every downstream
# derived rule.
#
# Dispatch is restricted statically to problems whose keyword type parameter is
# nonempty: a kwargs-free problem stays on Mooncake's own derived rules.
_problem_keyword_names(::Type) = ()
_problem_keyword_names(::Type{NamedTuple{K, T}}) where {K, T} = K
function _problem_keyword_names(
        ::Type{Base.Pairs{Symbol, V, I, T}},
    ) where {V, I, T <: NamedTuple{K, <:Any} where {K}}
    return _problem_keyword_names(T)
end

# The kwargs of an `AbstractODEProblem` carry their keyword names in the
# problem type's sixth type parameter (either a `@NamedTuple` or a
# `Base.Pairs{...,@NamedTuple{...}}`), which stays available for static
# dispatch on the concrete solve signature.
function _primitive_ode_solve(S::Type{<:Tuple})
    return !isempty(_problem_keyword_names(Base.unwrap_unionall(fieldtype(S, 4)).parameters[6]))
end

# Sequential rule: replay the whole `init`/`step!`/`solve!` flow once with the
# stage problems rebuilt from the sequence (first-stage overrides applied) and
# each stage solved through the keyword-bearing ODE primitive above. Because the
# stage solve is a registered primitive, Mooncake's derived rule for this replay
# contains no *nested* derived ODE rule -- building several distinct replay rules
# in one process therefore no longer trips Julia 1.12's LLVM `emit_unbox`
# codegen crash. The primitive moves the stage problem's solver controls onto
# the solve call, so the differentiated solve returns a keyword-free prob;
# `controls` re-attaches them to every later stage to keep the genuine
# tolerances and the primitive dispatch path.
function replayed_sequential_solve(problem, algorithm, options, forward)
    # The shared @generated helper keeps the split concrete; its generated code
    # slices by the problem's field names at compile time.
    seq_kwargs, kwargs = CorleoneBase.split_sequence_kwargs(problem, options)
    sequence = isempty(seq_kwargs) ? problem : SciMLBase.remake(problem; seq_kwargs...)
    stage1 = CorleoneBase.transition(sequence, nothing, 1)
    inner = CorleoneBase.prepare_stage_problem(SciMLBase.remake(stage1; kwargs...))
    # Solver controls carried by the rebuilt first stage; the genuine later
    # stages inherit them from `sol.prob`, so re-attach them below (the ODE
    # primitive's differentiated solve returns a keyword-free prob).
    controls = NamedTuple(inner.kwargs)
    save_flags1 = (dense = false, save_everystep = false, save_start = true, save_end = true)
    sol1 = CommonSolve.solve(inner, algorithm; save_flags1...)
    outputs = Vector([CorleoneBase.output(sequence, sol1, 1)])
    # Terminal-driven like the runtime; the typed index must not impose a
    # replay-only stage limit.
    terminal = CorleoneBase.terminal(sequence, outputs[1], 1)
    for stage in Iterators.countfrom(2)
        terminal && break
        next_problem = SciMLBase.remake(
            CorleoneBase.prepare_stage_problem(
                CorleoneBase.transition(sequence, outputs[stage-1], stage)
            );
            controls...
        )
        sol = CommonSolve.solve(next_problem, algorithm;
            save_start = false, save_end = true)
        push!(outputs, CorleoneBase.output(sequence, sol, stage))
        terminal = CorleoneBase.terminal(sequence, outputs[stage], stage)
    end
    return outputs
end

function Mooncake._is_primitive(
        ::Type{Mooncake.MinimalCtx}, ::Type{<:Mooncake.Mode}, ::Type{S}
    ) where {S <: Tuple{
        typeof(Core.kwcall), NamedTuple, typeof(CommonSolve.solve),
        SciMLBase.AbstractODEProblem, Any,
    }}
    return _primitive_ode_solve(S)
end

# Rebuild an `AbstractODEProblem` without its kwargs as a direct constructor
# call; differentiation flows through getfield like any projection.
kwargless_problem(problem::SciMLBase.ODEProblem) = SciMLBase.ODEProblem(
    problem.f, problem.u0, problem.tspan, problem.p, problem.problem_type,
)

# Rule for one genuine ODE solve whose problem carries keywords: replay it
# keyword-free with the controls splatted onto the solve call.
function replayed_ode_solve(problem, algorithm, options, forward)
    kwargless = kwargless_problem(problem)
    keywords = NamedTuple(problem.kwargs)
    return CommonSolve.solve(kwargless, algorithm; keywords..., options...)
end

function Mooncake.rrule!!(
        kwcall::CoDual{typeof(Core.kwcall)}, options::CoDual{<:NamedTuple},
        solve::CoDual{typeof(CommonSolve.solve)},
        problem::CoDual{<:SciMLBase.ODEProblem},
        algorithm::CoDual,
    )
    forward = CommonSolve.solve(primal(problem), primal(algorithm); primal(options)...)
    rule = Mooncake.build_rrule(
        replayed_ode_solve, primal(problem), primal(algorithm), primal(options), forward
    )
    result, back = rule(
        zero_fcodual(replayed_ode_solve), problem, algorithm, options, zero_fcodual(forward)
    )
    function ode_pullback!!(delta)
        _, dproblem, dalgorithm, doptions, _ = back(delta)
        return NoRData(), doptions, NoRData(), dproblem, dalgorithm
    end
    return result, ode_pullback!!
end

function Mooncake.frule!!(
        ::Dual{typeof(Core.kwcall)}, options::Dual{<:NamedTuple},
        ::Dual{typeof(CommonSolve.solve)},
        problem::Dual{<:SciMLBase.ODEProblem},
        algorithm::Dual,
    )
    forward = CommonSolve.solve(primal(problem), primal(algorithm); primal(options)...)
    rule = Mooncake.build_frule(
        replayed_ode_solve, primal(problem), primal(algorithm), primal(options), forward
    )
    result = rule(zero_dual(replayed_ode_solve), problem, algorithm, options, zero_dual(forward))
    return result
end


Mooncake.@is_primitive Mooncake.MinimalCtx Tuple{
    typeof(Core.kwcall), NamedTuple, typeof(CommonSolve.solve),
    CorleoneBase.AbstractSequentialProblem, Any,
}

function Mooncake.rrule!!(
        kwcall::CoDual{typeof(Core.kwcall)}, options::CoDual{<:NamedTuple},
        solve::CoDual{typeof(CommonSolve.solve)},
        problem::CoDual{<:CorleoneBase.AbstractSequentialProblem},
        algorithm::CoDual,
    )
    nt = primal(options)
    forward = CommonSolve.solve(primal(problem), primal(algorithm); nt...)
    rule = Mooncake.build_rrule(
        replayed_sequential_solve, primal(problem), primal(algorithm), nt, forward
    )
    result, back = rule(
        zero_fcodual(replayed_sequential_solve), problem, algorithm, options, zero_fcodual(forward)
    )
    function sequential_pullback!!(delta)
        _, dproblem, dalgorithm, doptions, _ = back(delta)
        return NoRData(), doptions, NoRData(), dproblem, dalgorithm
    end
    return result, sequential_pullback!!
end

# Forward mode reaches the same kwcall signature via the existing Parallel problem
# comment above: a MinimalCtx primitive is a DefaultCtx primitive too, so
# AutoMooncakeForward needs an frule for the sequential signature as well.
function Mooncake.frule!!(
        ::Dual{typeof(Core.kwcall)}, options::Dual{<:NamedTuple},
        ::Dual{typeof(CommonSolve.solve)},
        problem::Dual{<:CorleoneBase.AbstractSequentialProblem},
        algorithm::Dual,
    )
    nt = primal(options)
    forward = CommonSolve.solve(primal(problem), primal(algorithm); nt...)
    rule = Mooncake.build_frule(
        replayed_sequential_solve, primal(problem), primal(algorithm), nt, forward
    )
    result = rule(zero_dual(replayed_sequential_solve), problem, algorithm, options, zero_dual(forward))
    return result
end

end
