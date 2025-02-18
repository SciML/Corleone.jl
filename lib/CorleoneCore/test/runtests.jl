using CorleoneCore
using Test
using Aqua
using JET
using SafeTestsets

@testset "Code Quality" begin 
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(CorleoneCore)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(CorleoneCore; target_defined_modules = true)
    end
end

@safetestset "CorleoneCore.jl" begin
    @safetestset "Utilities" begin 
        include("utils.jl")
    end
    @safetestset "Single Parameter" begin
        include("grid_parameters.jl")
    end
    @safetestset "Parameter Container" begin 
        include("parameters.jl")
    end

    @safetestset "Grid Function" begin 
        include("grid_functions.jl")
    end
end
