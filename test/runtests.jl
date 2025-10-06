using Corleone
using Test
using Aqua
using JET
using SafeTestsets

@testset "Corleone.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Corleone)
    end
    #@testset "Code linting (JET.jl)" begin
    #    JET.test_package(Corleone; target_defined_modules=true)
    #end
    @testset "OED augmentation" begin
        include("augmentation.jl")
    end
end

# What to test?
# augmentation.jl:
#   - dimensions, expressions?,
#   - sorting of Fisher variables when sorting and applying symmetric_from_vector
# local_controls.jl:
#   - different constructors + functions, bounds, timegrids and so on
#   - construction of index_grid, get_subvector_indices -> Julius
# node_initialization.jl:
#   - test different init strategies on simple example
#   - for different layers (single / multiple shooting, OEDLayer, MultiExperimentLayer)
# multiple_shooting.jl:
#   - block structure of simple examples
# multi_experiments.jl:
#   - constructors
#   - inits, bounds, block structure
# general:
#   - prediction of layers: single shooting / multiple shooting / OED / MultiExperiment
#   - convergence? Lotka OC + Lotka OED

@generated function test_examples()
    expr = []
    example_dir = joinpath(@__DIR__, "examples")
    for f in readdir(example_dir)
        push!(expr, :(@safetestset $f begin
            include(joinpath($example_dir, $f))
        end))
    end
    return Expr(:block, expr...)
end

# Safetestset cannot interpolate. So we simply use a generator.
@testset "Examples" begin
    test_examples()
end
