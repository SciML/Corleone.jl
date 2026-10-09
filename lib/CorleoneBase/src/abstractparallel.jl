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
    get_problems(problem::AbstractParallelProblem)

Reserved interface hook for problem collections. No CorleoneBase implementation
calls it yet, so concrete parallel-problem types may leave it undefined.
"""
function get_problems end

"""
    prob_func(problem::AbstractParallelProblem, template, ctx)

Return the trajectory problem built from `template` and the ensemble context
`ctx`. The ensemble layer calls this as `prob_func(problem, template, ctx)` after
binding `problem`; `ctx` is a `SciMLBase.EnsembleContext`.
"""
function prob_func(::AbstractParallelProblem, template, ctx) end

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

    # Allowed keys as a static tuple of Symbols
    ensemble_keys = (:output_func, :reduction, :u_init, :safetycopy)

    # Keep only keys that actually exist in kwargs (Type-Stable via Base.structdiff)
    ensemblekwargs = Base.structdiff(nt_kwargs, Base.structdiff(nt_kwargs, NamedTuple{ensemble_keys}))

    # Or cleaner: extract ensemblekwargs directly using pure NamedTuple selection
    # ensemblekwargs = NamedTuple{intersect(keys(nt_kwargs), ensemble_keys)}(nt_kwargs)

    # solve_kwargs is simply the inverse diff
    solve_kwargs = Base.structdiff(nt_kwargs, ensemblekwargs)

    ensemble_problem = SciMLBase.EnsembleProblem(;
        prob = get_problem(problem),
        prob_func = Base.Fix1(prob_func, problem),
        ensemblekwargs...
    )

    return solve(ensemble_problem, alg, ensemblealg; trajectories, solve_kwargs...)
end
