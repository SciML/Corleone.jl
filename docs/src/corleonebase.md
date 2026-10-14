# [Sequential problems (CorleoneBase)](@id corleonebase)

`CorleoneBase` sequences SciML problems using CommonSolve. Controls, objectives,
bounds, and optimization belong in your application. See the
[manual single-shooting tutorial](@ref base_fishing) and [API reference](@ref base_api).

## Construction, output, and termination

Every hook takes the sequence first and may read `seq.parameters` and `seq.states`
(both default to `nothing`). `transition(seq, nothing, 1)` constructs the first
stage. Later calls receive the previous **stored output** and the next stage index.
Always supply a transition: the concrete constructor's default returns `nothing`,
not the wrapped problem. To use the template for stage one, return `seq.problem`.

`output(seq, solution, i)` runs after every solve, before storage; its default is
identity. `terminal(seq, stored_output, i)` stops when true; the concrete default
stops after one stage. Custom terminals own all stopping policies, including
failed solves. The abstract fallback instead stops at `i >= 1` or an unsuccessful
SciML return code. All stored outputs must fit the first output's buffer type.

State, parameters, and time intervals are never propagated automatically.
Remake the preceding solution's problem explicitly to carry an endpoint forward:

```jldoctest sequence
julia> using CorleoneBase, CommonSolve, OrdinaryDiffEqTsit5

julia> using SciMLBase: ODEProblem, remake, successful_retcode

julia> calls = Int[];

julia> template = ODEProblem((u, p, t) -> p .* u, [1.0], (0.0, 1.0), 0.0);

julia> sequence = SequentialProblem(template;
           transition = (seq, sol, i) -> begin
               push!(calls, i)
               i == 1 ? seq.problem : remake(sol.prob; u0 = sol.u[end],
                   tspan = (Float64(i - 1), Float64(i)))
           end,
           terminal = (seq, sol, i) -> i >= 3 || !successful_retcode(sol));

julia> stages = solve(sequence, Tsit5());

julia> (calls, length(stages), all(successful_retcode, stages))
([1, 2, 3], 3, true)

julia> (stages isa Vector, stages[end].prob.tspan, stages[end].u[end])
(true, (2.0, 3.0), [1.0])
```

## Sequence fields and stage keywords

`SciMLBase.remake(sequence; kwargs...)` replaces any of `problem`, `transition`,
`output`, `terminal`, `parameters`, and `states`, carrying the others over without
mutating the original sequence. `init` and `solve` route these field names to the
same sequence remake, so every hook observes the overrides.

All **remaining** keywords remake only the problem returned by
`transition(sequence, nothing, 1)`. There is no separate solver-keyword routing
or stored solver options. Later problems come exclusively from transition;
options can be embedded in their construction, or inherited by remaking `sol.prob`.
Stage one has hard-coded `dense = false`, `save_everystep = false`,
`save_start = true`, `save_end = true`. Later solves use `save_start = false`,
`save_end = true`; their other options come from the stage problem or solver defaults.
Thus later solutions need not contain their starting endpoint.

```jldoctest sequence-remake
julia> using CorleoneBase, CommonSolve, OrdinaryDiffEqTsit5

julia> using SciMLBase: ODEProblem, remake

julia> skeleton = SequentialProblem(
           ODEProblem((u, p, t) -> zero(u), [9.0], (0.0, 1.0), 0.0);
           transition = (seq, sol, i) -> remake(seq.problem;
               u0 = i == 1 ? seq.problem.u0 : sol.u[end],
               p = seq.parameters[i], tspan = (Float64(i - 1), Float64(i))),
           terminal = (seq, sol, i) -> i >= 3);

julia> stages = solve(skeleton, Tsit5(); parameters = [0.1, 0.2, 0.3], u0 = [2.0]);

julia> (skeleton.parameters, [sol.prob.p for sol in stages], stages[end].u[end])
(nothing, [0.1, 0.2, 0.3], [2.0])

julia> projected = remake(skeleton; parameters = zeros(3),
           transition = (seq, previous, i) -> seq.problem,
           output = (seq, sol, i) -> sol.u[end][1]);

julia> solve(projected, Tsit5())
3-element Vector{Float64}:
 9.0
 9.0
 9.0
```

## CommonSolve workflow and buffers

`init` already solves stage one. Its mutable iterator stores `problem`, bound
`transition`, `output`, `terminal`, `algorithm`, `buffer`, `current == 1`, and
`max_steps`. This is a CommonSolve state, not a streaming Julia iterator.
`step!` always attempts another stage and returns the iterator, **not a success
flag**. Guard manual stepping with `Base.isdone(it)` and your own failure policy.
`Base.isdone` calls only the terminal. `solve!` runs until that predicate fires.

```jldoctest sequence
julia> empty!(calls);

julia> it = init(sequence, Tsit5(); u0 = [2.0]);

julia> (it.current, it.max_steps, calls, Base.isdone(it))
(1, nothing, [1], false)

julia> step!(it) === it
true

julia> (it.current, calls, it.buffer[2].prob.u0)
(2, [1, 2], [2.0])

julia> completed = solve!(it);

julia> (completed === it.buffer, it.current, calls, length(completed))
(false, 3, [1, 2, 3], 3)

julia> solve!(it) === completed
false
```

`solve` and `solve!` return a new mutable vector containing a shallow copy of
the solved prefix `it.buffer[1:it.current]`. The iterator retains its capacity
and unused slots; the returned vector excludes them. Concrete abstract subtypes
can implement `has_size(problem) = true` and
`get_size(problem)::Int` to preallocate; their terminal must stop within that
capacity. The default is a growing buffer and `max_steps = nothing`.

There is no aggregate return-code accessor. For identity outputs use
`all(successful_retcode, stages)` and inspect individual solutions' `.retcode`,
end times, and expected count to establish full-horizon completion.
`ReturnCode.Terminated` counts as SciML success, but does not itself stop a custom
sequence. Exceptions from solvers and hooks propagate. No retry or penalty policy
is supplied for optimization objectives.

## Parallel problems

`ParallelProblem` wraps a template (often a `SequentialProblem`) for SciML ensembles.
Its `prob_func(problem, template, ctx)` constructs a trajectory using
`SciMLBase.EnsembleContext`, including `ctx.sim_id` and `ctx.rng`.

```julia
parallel = ParallelProblem(sequence; prob_func = (problem, template, ctx) ->
    remake(template; problem = remake(template.problem; u0 = rand(ctx.rng, 1))))
trajectories = solve(parallel, Tsit5(), EnsembleSerial(); trajectories = 8, seed = 1)
all(stages -> all(successful_retcode, stages), trajectories.u)
```

`trajectories` is required. `output_func`, `reduction`, `u_init`, and `safetycopy`
configure `SciMLBase.EnsembleProblem`; other options reach the ensemble solver and
trajectory solves. A sequential trajectory applies the stage-keyword rules above.
The ensemble hook uses a multi-argument closure on both Julia 1.10 and 1.12.

## Executed tutorial

The streamlined fishing example uses **120** intervals of length 0.1 over
`[0, 12]`. `fishing_stages` and `fishing_objective` are closures; the latter takes
the control vector and `fishing_sequence.states`. Controls live in sequence-wide
`parameters`, and the transition constructs stage one as well as every later
stage. It propagates all three states, including cumulative cost.

```sh
julia --startup-file=no --project=lib/CorleoneBase/examples/lotka_fishing -e 'using Pkg; Pkg.develop(path="lib/CorleoneBase"); Pkg.instantiate(); include("lib/CorleoneBase/examples/lotka_fishing/main.jl")'
```

The Docs group runs `docs/check_corleonebase.jl`: Literate executes the unmodified
tutorial, then Documenter checks exported documentation and doctests. Independent
tests check optimizer status, 120 stages, bounded controls, improvement, and
a scalar directional ForwardDiff derivative through construction and propagation
of every stage against a central difference. The derivative uses the full control
vector without another quadratic-cost full gradient.
