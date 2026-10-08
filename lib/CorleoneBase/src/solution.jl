"""
    SolutionWrapper

A read-only `AbstractVector` view over a
`SequentialProblemIterator`'s completed-stage buffer. It exposes exactly the
solved stages (`1:length`) and the exact `retcode` of the last solved stage. Unused
preallocated buffer slots are hidden: they never appear when indexing or iterating,
and they never influence `retcode`.

`CommonSolve.solve!` and therefore `CommonSolve.solve` return a `SolutionWrapper`;
`CommonSolve.init` keeps returning the iterator.
"""
struct SolutionWrapper{B, T} <: AbstractVector{T}
    "The underlying stage-solution buffer; ownership stays with the iterator."
    buffer::B
    "Number of solved stages exposed as `buffer[1:state]`."
    state::Int
end

SolutionWrapper(buffer::B, state::Int) where {B} = SolutionWrapper{B, eltype(B)}(buffer, state)

Base.size(w::SolutionWrapper) = (w.state,)
Base.IndexStyle(::Type{<:SolutionWrapper}) = IndexLinear()
Base.firstindex(w::SolutionWrapper) = 1
Base.lastindex(w::SolutionWrapper) = w.state

function Base.getindex(w::SolutionWrapper, i::Int)
    @boundscheck checkbounds(w, i)
    return w.buffer[i]
end

function Base.iterate(w::SolutionWrapper, i::Int = 1)
    i > w.state && return nothing
    return (w.buffer[i], i + 1)
end

"""
    retcode(w::SolutionWrapper)

The exact return code, of type `SciMLBase.ReturnCode.T`, of the last solved stage
exposed by `w`. For a wrapper whose last stage failed, that is the failure's exact
return code; unused preallocated slots never contribute.
"""
retcode(w::SolutionWrapper) = w.buffer[w.state].retcode

SciMLBase.successful_retcode(w::SolutionWrapper) = SciMLBase.successful_retcode(retcode(w))
