"""
    SequentialProblem(problem; transition = (seq, sol, i) -> nothing,
                      output = (seq, sol, i) -> sol,
                      terminal = (seq, sol, i) -> i >= 1,
                      parameters = nothing, states = nothing)

Solve SciML stages with a shared algorithm. All hooks receive the sequence first
and may read its optional `parameters` and `states`. Construction and propagation
are explicit: `transition(seq, nothing, 1)` constructs stage one; subsequent calls
receive the preceding stored output and the next index. Supply a transition even
for a single-stage solve: the constructor's default returns `nothing`.

`output(seq, solution, i)` transforms each solved stage before storage (identity
by default). Both transition and terminal receive these transformed values.
All outputs must fit the buffer's first-output element type.
`terminal(seq, stored_output, i)` stops the sequence when true. The concrete
default stops at stage one; a custom terminal must implement any desired failure
policy, for example `i >= n || !SciMLBase.successful_retcode(sol)`.

`CommonSolve.init` immediately solves stage one and returns a mutable
[`SequentialProblemIterator`](@ref). `step!` advances it and returns the iterator,
not a success flag; it checks neither termination nor prior success. `solve!` and
`solve` return a new mutable vector containing the solved outputs, a shallow
copy of `it.buffer[1:it.current]`. The iterator retains unused buffer capacity.
Inspect return codes on individual SciML solutions, not the
buffer. Solver and callback exceptions propagate.

Keywords naming sequence fields (`problem`, `transition`, `output`, `terminal`,
`parameters`, `states`) remake the sequence without mutating it. All remaining
keywords remake only the stage built by `transition(sequence, nothing, 1)`;
they are not stored or forwarded to later solves. Solver options can be carried
by the stage problems themselves. Stage one is solved with `dense = false`,
`save_everystep = false`, `save_start = true`, `save_end = true`; later stages
use `save_start = false`, `save_end = true`. Supply an eventual terminal condition.
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
`output`, `terminal`, `parameters`, `states`) may be replaced; the returned sequence is a
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
