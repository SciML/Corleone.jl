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
    @testset "Optimal control" begin
        include("oc.jl")
    end
    @testset "OED augmentation" begin
        include("augmentation.jl")
    end
    @testset "Multiple shooting" begin
        include("multiple_shooting.jl")
    end
end

# What to test?
# local_controls.jl:
#   - different constructors + functions, bounds, timegrids and so on
#   - construction of index_grid, get_subvector_indices -> Julius
# node_initialization.jl:
#   - for different layers (single / multiple shooting, OEDLayer, MultiExperimentLayer)
# multi_experiments.jl:
#   - constructors
#   - inits, bounds, block structure
# general:
#   - prediction of layers: single shooting / multiple shooting / OED / MultiExperiment
#   - convergence? Lotka OC + Lotka OED

# using Coverage;
#coverage = process_folder();
#coverage = merge_coverage_counts(coverage, filter!(
#    let prefixes = (joinpath(pwd(), "src", ""),)
#        c -> any(p -> startswith(c.filename, p), prefixes)
#    end,
#LCOV.readfolder("test")));
#covered_lines, total_lines = get_summary(coverage);
#println("Coverage $(covered_lines / total_lines)");

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
