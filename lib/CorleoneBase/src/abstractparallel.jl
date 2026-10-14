"""
    AbstractParallelProblem

Supertype for CorleoneBase wrappers that fan a problem out across a SciML
[`SciMLBase.EnsembleProblem`]. Concrete subtypes provide

  - `get_problem(problem::AbstractParallelProblem)`: the shared template problem
    every trajectory starts from, and
  - `prob_func(problem::AbstractParallelProblem, template, ctx)`: build one
    trajectory's problem from `template` and the ensemble context `ctx`.

`ctx` is a `SciMLBase.EnsembleContext` exposing `ctx.sim_id` (`1:trajectories`),
`ctx.repeat`, `ctx.rng`, `ctx.sim_seed`, and `ctx.master_rng`, mirroring
SciMLBase's `prob_func(prob, ctx)` contract.

A parallel problem is not a `SciMLBase.AbstractSciMLProblem`; solve it with
`CommonSolve.solve(problem, algorithm, ensemblealg; trajectories, kwargs...)`,
which constructs and solves a `SciMLBase.EnsembleProblem`.
"""
abstract type AbstractParallelProblem end

"""
    get_problem(problem::AbstractParallelProblem)

Return the template `SciMLBase.AbstractSciMLProblem` that `prob_func` is called
on for every trajectory.
"""
function get_problem(::AbstractParallelProblem) end

"""
    prob_func(problem::AbstractParallelProblem, template, ctx)

Return the trajectory problem built from `template` and the ensemble context
`ctx`. The ensemble layer calls this as `prob_func(problem, template, ctx)` after
binding `problem`; `ctx` is a `SciMLBase.EnsembleContext`.
"""
function prob_func(::AbstractParallelProblem, template, ctx) end

# Keywords that configure a `SciMLBase.EnsembleProblem` rather than its solve.
const ENSEMBLE_KEYS = (:output_func, :reduction, :u_init, :safetycopy)

"""
    fix_me(f, x) -> callable

EnsembleProblem convenience wrapper: turn a CorleoneBase three-argument hook
`f(problem, template, ctx)` into the two-argument `prob_func(prob, i|ctx)`
callable a `SciMLBase.EnsembleProblem` expects, bound with `x`. On Julia >= 1.12
this is `Base.Fix1(f, x)`; on 1.10/1.11 `Base.Fix1` only forwards one remaining
argument, so an equivalent multi-argument closure is used instead.
"""
@static if VERSION >= v"1.12"
    fix_me(f, x) = Base.Fix1(f, x)
else
    fix_me(f, x) = let f = f, x = x
        (args...) -> f(x, args...)
    end
end

"""
    select_keys(nt::NamedTuple, keys) -> NamedTuple

Return the members of `keys` that occur in `nt`, as a NamedTuple carrying
exactly the selected keys (in `keys` order). A type-stable replacement for the
double `Base.structdiff` select.
"""
function select_keys(nt::NamedTuple, keys::Tuple{Vararg{Symbol}})
    return Base.structdiff(nt, Base.structdiff(nt, NamedTuple{keys}))
end

"""
    ensemble_problem(problem::AbstractParallelProblem; kwargs...) -> SciMLBase.EnsembleProblem

Build the `SciMLBase.EnsembleProblem` for a parallel problem: the template from
`get_problem`, the problem's `prob_func` hook bound with `fix_me`, plus any
ensemble keywords (`output_func`, `reduction`, `u_init`, `safetycopy`).
"""
function ensemble_problem(problem::AbstractParallelProblem; kwargs...)
    return SciMLBase.EnsembleProblem(;
        prob = get_problem(problem),
        prob_func = fix_me(prob_func, problem),
        kwargs...
    )
end

"""
    solve(problem::AbstractParallelProblem, alg::SciMLBase.AbstractSciMLAlgorithm,
          ensemblealg::SciMLBase.EnsembleAlgorithm; trajectories, kwargs...)

Solve `problem` as a SciML ensemble. `trajectories` (required) is the number of
ensemble members. `prob_func` is bound to `problem` and handed to a
`SciMLBase.EnsembleProblem` built from `get_problem(problem)`.

Keywords named `output_func`, `reduction`, `u_init`, or `safetycopy` configure
that `SciMLBase.EnsembleProblem`; every other keyword is forwarded to
`SciMLBase.solve`, so ensemble keywords such as `rng`, `seed`, and `batch_size`
and ordinary solver keywords such as `abstol` and `maxiters` keep their usual
meaning. Under the default `output_func` the result is a
`SciMLBase.EnsembleSolution` holding one entry per trajectory.
"""
@inline function CommonSolve.solve(
        problem::T, alg::SciMLBase.AbstractSciMLAlgorithm, ensemblealg::SciMLBase.EnsembleAlgorithm;
        trajectories,
        kwargs...
    ) where {T <: AbstractParallelProblem}

    # Convert kwargs to a NamedTuple (type-stable)
    nt_kwargs = NamedTuple(kwargs)

    # Keywords named for the EnsembleProblem configure it; the rest reach solve.
    ensemble_kwargs = select_keys(nt_kwargs, ENSEMBLE_KEYS)
    solve_kwargs = Base.structdiff(nt_kwargs, ensemble_kwargs)

    ensembles = ensemble_problem(problem; ensemble_kwargs...)

    return solve(ensembles, alg, ensemblealg; trajectories, solve_kwargs...)
end
