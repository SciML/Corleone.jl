# [Sequential problems (CorleoneBase)](@id corleonebase)

`CorleoneBase` is a small, independently usable sublibrary. It sequences SciML
problems using CommonSolve; it does not construct controls, objectives, bounds,
optimization problems, or plots. Those belong in your application. See
[manual single shooting](@ref base_fishing) for a runnable fishing example and
[the API reference](@ref base_api) for the public result and problem types.

## Callback contract and propagation

Construct `SequentialProblem(first_problem; transition, terminal)`. With defaults,
only the first problem is solved. To continue, supply both callbacks:

* `terminal(sol, i)` receives the solution and **current** stage index (first is
  1). Returning `true` stops before constructing another stage.
* `transition(sol, i)` receives the previous solution and **next** stage index
  (first call is 2). Return a SciML problem compatible with the same algorithm
  and stored solution buffer type.

There is no automatic propagation of state, parameters, or time intervals.
Typically `remake(sol.prob; u0 = sol.u[end], p = ..., tspan = ...)` carries the
endpoint to the next stage. Copy or replace state explicitly if the next stage
must reset it. You must ensure an eventual terminal condition; there is no outer
stage limit. A stage solver's `maxiters` limits that stage, not the sequence.

```jldoctest sequence
julia> using CorleoneBase, CommonSolve, OrdinaryDiffEqTsit5

julia> using SciMLBase: ODEProblem, remake, ReturnCode, successful_retcode

julia> calls = Int[];

julia> first_problem = ODEProblem((u, p, t) -> p .* u, [1.0], (0.0, 1.0), 0.0);

julia> sequence = SequentialProblem(first_problem;
           transition = (sol, i) -> begin
               push!(calls, i)
               remake(sol.prob; u0 = sol.u[end], p = 0.0,
                      tspan = (Float64(i - 1), Float64(i)))
           end,
           terminal = (sol, i) -> i >= 3);

julia> stages = solve(sequence, Tsit5(); abstol = 1e-9, reltol = 1e-9,
                     save_everystep = false);

julia> (calls, length(stages), all(successful_retcode, stages))
([2, 3], 3, true)

julia> (stages[1].prob.tspan, stages[end].prob.tspan, stages[end].u[end])
((0.0, 1.0), (2.0, 3.0), [1.0])

julia> (stages isa SolutionWrapper, CorleoneBase.retcode(stages) == ReturnCode.Success,
        successful_retcode(stages))
(true, true, true)
```

## CommonSolve workflow and keywords

`solve(sequence, algorithm; kwargs...)` initializes and completes the sequence.
`init` **already solves the first stage**, returning a mutable iterator with
`state == 1`, the solved-stage `buffer`, and the algorithm and remaining solve
keywords. It is a CommonSolve state object, not a Julia iteration protocol for
streaming stage solves: use `step!`, `Base.isdone`, and `solve!` as below.

`u0`, `p`, and `tspan` keywords remake **only the initial problem**. They are
removed from the stored solve keywords. All other keywords (for example
`abstol`, `reltol`, `save_everystep`, `save_start`, `save_end`, `maxiters`) are
forwarded unchanged to **each** stage solver, whose own semantics apply.
Subsequent problems come exclusively from `transition`; overrides are not
reapplied, although a transition remaking `sol.prob` can intentionally inherit
the first stage's parameters. A shared absolute `saveat` schedule may be
inappropriate for different stage spans. Propagating endpoints requires that
they actually be saved (keep `save_end = true`).

```jldoctest sequence
julia> empty!(calls);

julia> it = init(sequence, Tsit5(); u0 = [2.0], p = 0.0, tspan = (0.0, 1.0),
                abstol = 1e-9, reltol = 1e-9, save_everystep = false);

julia> (it.state, isempty(calls), it.buffer[1].u[end], Base.isdone(it))
(1, true, [2.0], false)

julia> step!(it)  # explicitly solves stage 2; returns its success flag
true

julia> (it.state, calls, it.buffer[2].prob.u0)
(2, [2], [2.0])

julia> completed = solve!(it);

julia> (it.state, calls, Base.isdone(it), length(completed), completed[end].u[end])
(3, [2, 3], true, 3, [2.0])

julia> length(solve!(it))  # a completed successful iterator does not advance again
3

julia> first_problem.u0  # original template was not remade in place
1-element Vector{Float64}:
 1.0
```

`step!` is deliberately low level: it always attempts a next stage, without
checking the terminal predicate or the previous stage's success. Guard it with
`successful_retcode(it.buffer[it.state]) && !Base.isdone(it)` when manually
stepping. `Base.isdone` calls only the terminal predicate; it does not signify
successful integration or failure. Prefer `solve!` for automatic stopping on
either terminal completion or stage failure.

## Results and failure behavior

`solve` and `solve!` return a read-only `SolutionWrapper <: AbstractVector` over
the iterator's retained stage solutions. Use `length`, indexing, `end`, and
iteration; `stages[i]` is the original stage solution, with its usual `.u`, `.t`,
`.prob`, and `.retcode`. Only solved slots are exposed. The wrapper shares the
buffer, does not deep-copy solutions, and stores its length at creation; it is
not an immutable snapshot of the underlying data. Do not mutate the buffer.

`CorleoneBase.retcode(stages)` (qualified, not exported) is the **exact last
retained stage's** return code, not a new aggregated code or an optimization
status. `SciMLBase.successful_retcode(stages)` checks that code. In an ordinary
`solve`, preceding retained stages succeeded; inspect individual stages, their
end times, and the expected count if your application needs full-horizon
completion. A terminal predicate can intentionally stop early with a successful
result.

Stage success follows SciML's return-code classification: `ReturnCode.Terminated`
is successful, even if an integration callback stopped before the stage's final
time. It does **not** automatically terminate the sequence. Use the sequence's
`terminal` predicate to request that policy. Here the forwarded solver callback
stops both stages early, but the sequence still reaches its two-stage limit:

```jldoctest
julia> using CorleoneBase, CommonSolve, OrdinaryDiffEqTsit5

julia> using SciMLBase: ODEProblem, remake, ReturnCode, successful_retcode,
                       DiscreteCallback, terminate!

julia> stopped_sequence = SequentialProblem(
           ODEProblem((u, p, t) -> zero(u), [1.0], (0.0, 1.0));
           transition = (sol, i) -> remake(sol.prob; u0 = sol.u[end],
               tspan = (sol.t[end], sol.t[end] + 1.0)),
           terminal = (sol, i) -> i >= 2);

julia> stop_stage = DiscreteCallback((u, t, integrator) -> true, terminate!);

julia> stopped = solve(stopped_sequence, Tsit5(); callback = stop_stage,
                      adaptive = false, dt = 0.25);

julia> (length(stopped), successful_retcode(stopped),
        all(sol -> sol.retcode == ReturnCode.Terminated, stopped),
        all(sol -> sol.t[end] < sol.prob.tspan[2], stopped))
(2, true, true, true)
```

An unsuccessful first solve is retained as a one-stage failure result and
`solve!` never transitions. A later unsuccessful stage is retained too, then
`solve!` stops without calling another transition. `step!` returns `false` for
that failed next stage, after storing it and incrementing `state`. A failed
iterator is not automatically retried by a subsequent `solve!`. Exceptions from
solvers or callbacks propagate; they are not converted to return codes. There
is no built-in penalty, retry, or fallback policy for an optimization objective.

This example silences the intentional solver warning with Julia's standard
logger, while checking the failure code and absence of transitions.

```jldoctest
julia> using CorleoneBase, CommonSolve, OrdinaryDiffEqTsit5

julia> using SciMLBase: ODEProblem, remake, ReturnCode, successful_retcode

julia> using Logging: with_logger, NullLogger

julia> transitions = Int[];

julia> failing = SequentialProblem(
           ODEProblem((u, p, t) -> -u, [1.0], (0.0, 1.0));
           transition = (sol, i) -> (push!(transitions, i); sol.prob),
           terminal = (sol, i) -> i >= 3);

julia> failed = with_logger(NullLogger()) do
           solve(failing, Tsit5(); maxiters = 0)
       end;

julia> (length(failed), isempty(transitions), successful_retcode(failed),
        CorleoneBase.retcode(failed) == ReturnCode.MaxIters)
(1, true, false, true)
```

## Running and checking the tutorial

From the repository root, run the standalone Literate source with its own
environment and the checked-out sublibrary (no registered CorleoneBase needed):

```sh
julia --startup-file=no --project=lib/CorleoneBase/examples/lotka_fishing -e 'using Pkg; Pkg.develop(path="lib/CorleoneBase"); Pkg.instantiate(); include("lib/CorleoneBase/examples/lotka_fishing/main.jl"); display(report)'
```

The assertions check local optimizer success, all 24 stage solves, finite
results, bounded controls, and improvement over constant intensity 0.5, including
a tighter-tolerance reevaluation. The output reports numerical tolerances and
termination, not global optimality. Optimization, AD, and solver dependencies
are confined to the example/docs/test environments, not runtime dependencies.

`julia --startup-file=no --project=docs docs/make.jl` builds the full shared manual
from the repository root. For a focused strict render/doctest/tutorial check of
these same pages, see `docs/check_corleonebase.jl` and its setup instructions.
