using Corleone
using Test
using Aqua
using JET
using SafeTestsets
#@testset "Corleone.jl" begin
#    @testset "Code quality (Aqua.jl)" begin
#        Aqua.test_all(Corleone)
#    end
#    @testset "Code linting (JET.jl)" begin
#        JET.test_package(Corleone; target_defined_modules = true)
#    end
#    # Write your tests here.
#end

@generated function test_examples()
    expr = []
    example_dir = joinpath(@__DIR__, "examples")
    for f in readdir(example_dir)
        push!(expr, :(@safetestset $f begin include(joinpath($example_dir, $f)) end))
    end
    return Expr(:block, expr...)
end 
 
# Safetestset cannot interpolate. So we simply use a generator. 
@testset "Examples" begin test_examples() end 
