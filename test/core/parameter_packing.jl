using Test, Corleone, ComponentArrays, ForwardDiff, ReverseDiff, Zygote
using SciMLBase, OrdinaryDiffEqTsit5, LuxCore, Random

# A different reconstruction result verifies that the bridge only needs named
# parameter access, rather than a specific container type.
struct PropertyParameters{I, P}
    initialcondition::I
    parameter::P
end
struct PropertyReconstructor{R}
    reconstruct::R
end
function (r::PropertyReconstructor)(q)
    ps = r.reconstruct(q)
    return PropertyParameters(ps.initialcondition, ps.parameter)
end
function Corleone.pack(::Val{:TestPacking}, ps::NamedTuple)
    q, reconstruct = pack(Val(:ComponentArrays), ps)
    return q, PropertyReconstructor(reconstruct)
end

@testset "Parameter packing and view reconstruction" begin
    tag = Val(:ComponentArrays)
    @test default_vectorizer() === tag
    ps = (; a = [1.0, 2.0], nested = (; b = reshape([3.0, 4.0], 1, 2), empty = (;)))
    q, reconstruct = pack(tag, ps)
    tree = @inferred reconstruct(q)
    @test q isa Vector{Float64}
    @test tree isa ComponentVector
    @test tree.nested isa ComponentVector
    @test tree.a == ps.a && tree.nested.b == ps.nested.b
    @test isempty(tree.nested.empty)
    @test size(tree.nested.b) == (1, 2)
    @test first(pack(tag, tree)) === q
    @test getdata(tree) === q
    @test Base.mightalias(tree.a, q)
    @test Base.mightalias(tree.nested.b, q)
    q[1] = 5.0
    @test tree.a[1] == 5.0
    @test ps.a[1] == 1.0 # Initial packing allocated independent flat storage.
    current = copy(q)
    current .+= 1
    reconstructed = @inferred reconstruct(current)
    @test getdata(reconstructed) === current
    @test reconstructed.a == current[1:2]
    @test Base.mightalias(reconstructed.nested.b, current)
    empty, restore_empty = pack(tag, (;))
    @test isempty(empty) && isempty(restore_empty(empty))
    @test_throws DimensionMismatch reconstruct(q[1:2])
    @test_throws ArgumentError pack(tag, (; a = [1.0 + im]))
    loss = x -> begin
        p = reconstruct(x)
        sum(abs2, p.a) + sum(abs2, p.nested.b)
    end
    for gradient in (ForwardDiff.gradient, ReverseDiff.gradient, (f, x) -> only(Zygote.gradient(f, x)))
        @test gradient(loss, current) ≈ 2current
    end
end

@testset "SciMLBase bridge with interchangeable packing" begin
    # This file imports SciMLBase, not Optimization.jl.
    layer = ShootingLayer(
        ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 1.0), [0.3]), Tsit5(),
        Tunable(1; kind = Val(:parameter), default = 0.3, lowerbound = 0.0, upperbound = 1.0) => nothing
    )
    ps, st = LuxCore.setup(MersenneTwister(1), layer)
    model = CorleoneDynamicOptProblem(layer, :u₁; ps, st)
    ad = SciMLBase.ADTypes.AutoForwardDiff()
    for tag in (Val(:ComponentArrays), Val(:TestPacking))
        problem = SciMLBase.OptimizationProblem(model, ad; vectorize = tag)
        q = collect(problem.u0)
        @test problem.p === st
        @test collect(problem.lb) == [0.0] && collect(problem.ub) == [1.0]
        @test (@inferred problem.f.f(q, st)) ≈ 0.3
        @test ForwardDiff.gradient(x -> problem.f.f(x, st), q) ≈ [1.0]
        @test problem.f.f([0.6], st) ≈ 0.6
        f = SciMLBase.OptimizationFunction(model, ad; vectorize = tag)
        @test f.f(q, st) ≈ 0.3
    end
end

@testset "Vectorizer extension selection in a fresh process" begin
    script = joinpath(@__DIR__, "vectorizer_selection_process.jl")
    project = dirname(Base.active_project())
    @test success(`$(Base.julia_cmd()) --startup-file=no --project=$project $script`)
end
