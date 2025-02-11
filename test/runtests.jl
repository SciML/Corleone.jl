using Corleone
using Test
using Aqua
using JET

@testset "Corleone.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(Corleone)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(Corleone; target_defined_modules = true)
    end
    # Write your tests here.
end
