"""
    SequentialProblem(problem; transition = (seq, sol, i) -> nothing,
                      terminal = (seq, sol, i) -> i >= 1,
                      parameters = nothing, states = nothing)

Solve a sequence of SciML problems with `CommonSolve.init`/`solve` and an algorithm.
The terminal predicate is called as `terminal(seq, sol, i)` with the sequence
itself, the current stage's solution, and the 1-based current index. If it is
`false`, `transition(seq, sol, next_index)` supplies the next SciML problem,
starting at index 2. Both callbacks must accept the sequence as their first
argument and can read the sequence-wide `seq.parameters` and `seq.states`
fields from it; legacy two-argument `(sol, i)` callables are not supported.
State, parameters, and time spans are not propagated automatically: use e.g.
`remake(sol.prob; u0 = sol.u[end], tspan = ..., p = ...)` in the transition.
Returned problems must support the same algorithm and their solutions must be
compatible with the iterator's solution buffer element type.

`parameters` and `states` are optional sequence-wide fields, `nothing` by
default. They are plain data for the callbacks: neither is applied to any
stage problem automatically, and the first stage's state, parameters, and time
span come exclusively from its explicit construction (the wrapped problem plus
first-stage `init` keywords such as `u0`, `p`, `tspan`). A new sequence with
either field replaced is built by `SciMLBase.remake(sequence; parameters = ...,
states = ...)` — any field (`problem`, `transition`, `terminal`, `parameters`,
`states`) can be replaced, the others are carried over. `CommonSolve.init` and
`CommonSolve.solve` accept the same names as keywords: keywords naming a field
of the sequence remake the sequence before solving, so the callbacks observe
the overrides while the original sequence object is never mutated.

`CommonSolve.init` solves the first stage immediately. `CommonSolve.step!` solves
one next stage and returns its success flag; it does not check termination or
previous-stage success. `CommonSolve.solve!` runs until the terminal predicate
returns true or a stage fails, retaining the failed stage. An unsuccessful first
stage never transitions during `solve!`. Solver and callback exceptions propagate.
Supply a terminal condition that eventually stops the sequence; the default
stops after one stage and the default transition supplies no next problem.

The completed
iterator stores stage solutions in `buffer` and the current index in `state`.
`CommonSolve.solve!` and `CommonSolve.solve` return a read-only `SolutionWrapper`
(an `AbstractVector`) over the solved stages, whose `retcode` is the last solved
stage's exact return code; `CommonSolve.init` returns the iterator itself.
Initial keywords that name a field of the initial problem (for an `ODEProblem`:
`u0`, `p`, `tspan`, `f`, `kwargs`, `problem_type`) remake only the first stage,
which `prepare_stage_problem` then normalizes; remaining keywords are
forwarded unchanged to every stage. Sequence-field keywords are routed before
problem-field keywords, so a wrapped problem with a same-named field cannot
capture them. Keep endpoint saving enabled when propagating
`sol.u[end]`. `maxiters` limits each stage's solver, not the total number of
stages.
"""
struct SequentialProblem{P, T, O, R, X, S} <: AbstractSequentialProblem
    "The initial problem to solve (first stage)"
    problem::P
    "Transition callable that receives `(sequence, solution, next_index)`
  and returns the next `AbstractSciMLProblem`"
    transition::T
    "Output function"
    output::O
    "Terminal predicate callable that receives `(sequence, solution, current_index)`
  and returns `Bool`. When `true`, no further stages are solved."
    terminal::R
    "Sequence-wide parameters the callbacks can read as `seq.parameters`;
  `nothing` by default and never applied to a stage problem automatically"
    parameters::X
    "Sequence-wide states the callbacks can read as `seq.states`;
  `nothing` by default and never applied to a stage problem automatically"
    states::S
end

get_problem(x::SequentialProblem) = x.problem
transition(x::SequentialProblem, sol, i) = x.transition(x, sol, i)
terminal(x::SequentialProblem, sol, i) = x.terminal(x, sol, i)
output(x::SequentialProblem, sol, i) = x.output(x, sol, i)

function SequentialProblem(
        problem::SciMLBase.AbstractSciMLProblem;
        transition = (seq, sol, i) -> nothing,
        output = (seq, sol, i) -> sol,
        terminal = (seq, sol, i) -> i >= 1,
        parameters = nothing,
        states = nothing,
    )
    return SequentialProblem{typeof(problem), typeof(transition), typeof(output), typeof(terminal), typeof(parameters), typeof(states)}(
        problem, transition, output, terminal, parameters, states
    )
end

"""
    SciMLBase.remake(sequence::SequentialProblem; kwargs...) -> SequentialProblem

Return a new sequence with the given fields replaced and the remaining fields
carried over unchanged. Every field of the sequence (`problem`, `transition`,
`terminal`, `parameters`, `states`) may be replaced; the returned sequence is a
new object and the input is never mutated.
"""
function SciMLBase.remake(sequence::SequentialProblem; kwargs...)
    nt = NamedTuple(kwargs)
    return SequentialProblem(
        haskey(nt, :problem) ? nt.problem : sequence.problem,
        haskey(nt, :transition) ? nt.transition : sequence.transition,
        haskey(nt, :output) ? nt.output : sequence.output,
        haskey(nt, :terminal) ? nt.terminal : sequence.terminal,
        haskey(nt, :parameters) ? nt.parameters : sequence.parameters,
        haskey(nt, :states) ? nt.states : sequence.states,
    )
end
