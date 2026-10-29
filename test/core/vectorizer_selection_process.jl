# Run in a fresh process: extensions cannot be unloaded once another test loaded them.
using Test, Corleone, SciMLBase, OrdinaryDiffEqTsit5

layer = ShootingLayer(ODEProblem((u, p, t) -> [p[1]], [0.0], (0.0, 1.0), [0.3]), Tsit5())
model = CorleoneDynamicOptProblem(layer, :u₁)
ad = SciMLBase.ADTypes.AutoForwardDiff()

@testset "No loaded vectorization extension" begin
    @test Base.get_extension(Corleone, :CorleoneComponentArraysExtension) === nothing
    err = try
        default_vectorizer()
    catch e
        e
    end
    @test err isa UndefKeywordError
    @test err.var === :vectorize
    @test_throws UndefKeywordError SciMLBase.OptimizationFunction(model, ad)
    @test_throws UndefKeywordError SciMLBase.OptimizationProblem(model, ad)
    @test_throws UndefKeywordError SciMLBase.OptimizationProblem(layer, ad)
end

using ComponentArrays

@testset "Default selection after loading an extension" begin
    @test default_vectorizer() === Val(:ComponentArrays)
    problem = SciMLBase.OptimizationProblem(model, ad)
    @test problem.u0 isa Vector{Float64}
    @test problem.f.f(problem.u0, problem.p) ≈ 0.3
    @test SciMLBase.OptimizationFunction(model, ad).f(problem.u0, problem.p) ≈ 0.3
end
