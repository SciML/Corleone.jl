"""
    ParallelProblem(problem; prob_func = (problem, template, ctx) -> template)

Wrap a SciML problem — commonly a [`SequentialProblem`](@ref) — so it can be
solved as a SciML ensemble. `problem` is the template; each of the
`trajectories` requested by `CommonSolve.solve` runs
`prob_func(problem, template, ctx)` to build the trajectory-specific problem.
`ctx` is a `SciMLBase.EnsembleContext` exposing `ctx.sim_id`
(`1:trajectories`), `ctx.repeat`, `ctx.rng`, `ctx.sim_seed`, and
`ctx.master_rng`; use it for per-trajectory variation such as random initial
conditions. The returned problem must have the same concrete type as `template`.

Solve it with `CommonSolve.solve(problem, algorithm, ensemblealg; trajectories,
kwargs...)`. Keywords `output_func`, `reduction`, `u_init`, and `safetycopy`
configure the underlying `SciMLBase.EnsembleProblem`; all other keywords reach
the ensemble and inner solves.

The default `prob_func` ignores the context and returns the template unchanged,
so every trajectory solves the identical problem.
"""
struct ParallelProblem{P, F} <: AbstractParallelProblem
    "The template problem each trajectory starts from."
    problem::P
    "Trajectory hook `(problem, template, ctx) -> problem′`."
    prob_func::F
end

"""
    get_problem(problem::ParallelProblem)

Return the template problem wrapped by `problem`.
"""
get_problem(x::ParallelProblem) = x.problem

"""
    prob_func(problem::ParallelProblem, template, ctx)

Call the hook stored in `problem` with `(problem, template, ctx)` and return the
resulting trajectory problem.
"""
prob_func(x::ParallelProblem, template, ctx) = x.prob_func(x, template, ctx)

"""
    ParallelProblem(problem; prob_func = (problem, template, ctx) -> template)

Keyword constructor; see the `ParallelProblem` type documentation for the hook
contract.
"""
ParallelProblem(problem; prob_func = (x, prob, i) -> prob) =
    ParallelProblem(problem, prob_func)
