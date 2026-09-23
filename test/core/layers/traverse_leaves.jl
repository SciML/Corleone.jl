# ---------------------------------------------------------------------------
# traverse_leaves.jl – unit tests for the shared Layer/WrapperLayer/
# ContainerLayer/NamedTuple/Tuple traversal combinator (issue #3)
# ---------------------------------------------------------------------------

struct _DummyLeaf <: LuxCore.AbstractLuxLayer end

struct _DummyWrapper{L} <: LuxCore.AbstractLuxWrapperLayer{(:inner,)}
    inner::L
end

struct _DummyContainer{N <: NamedTuple} <: LuxCore.AbstractLuxContainerLayer{(:children,)}
    children::N
end

_dummy_double(::_DummyLeaf, ps, st) = 2ps

@testset "traverse_leaves – leaf delegates directly to f" begin
    @test Corleone.traverse_leaves(_dummy_double, _DummyLeaf(), 5, nothing) == 10
end

@testset "traverse_leaves – wrapper unwraps one level via getproperty" begin
    # AbstractLuxWrapperLayer convention: ps/st mirror the inner layer's own
    # shape directly (unlike containers), so they pass through unchanged.
    wrapped = _DummyWrapper(_DummyLeaf())
    @test Corleone.traverse_leaves(_dummy_double, wrapped, 5, nothing) == 10
end

@testset "traverse_leaves – container recurses per name, preserves shape" begin
    container = _DummyContainer((; a = _DummyLeaf(), b = _DummyLeaf()))
    ps = (; children = (; a = 3, b = 4))
    st = (; children = (; a = nothing, b = nothing))
    result = Corleone.traverse_leaves(_dummy_double, container, ps, st)
    @test result.children.a == 6
    @test result.children.b == 8
end

@testset "traverse_leaves – works through ComponentArrays ps/st (getproperty, not getfield)" begin
    using ComponentArrays

    container = _DummyContainer((; a = _DummyLeaf(), b = _DummyLeaf()))
    ps = ComponentArray(; children = (; a = 3.0, b = 4.0))
    st = (; children = (; a = nothing, b = nothing))
    result = Corleone.traverse_leaves(_dummy_double, container, ps, st)
    @test result.children.a == 6.0
    @test result.children.b == 8.0
end

@testset "traverse_leaves – tuple of leaves recurses elementwise" begin
    leaves = (_DummyLeaf(), _DummyLeaf())
    ps = (3, 4)
    st = (nothing, nothing)
    @test Corleone.traverse_leaves(_dummy_double, leaves, ps, st) == (6, 8)
end
