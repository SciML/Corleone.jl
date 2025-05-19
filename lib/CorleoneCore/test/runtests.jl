using CorleoneCore
using Test
using Aqua

@testset "Code Quality" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(CorleoneCore,
            ambiguities = (; recursive = false)
        )
    end
end


#@testset "Lotka MS" begin
#    include("./lotka_MS.jl")
#end
