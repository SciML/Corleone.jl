using Test
using JET
using SafeTestsets

@testset "Corleone.jl" begin
    @safetestset "Code quality (Aqua.jl)" begin
        using Aqua
        using Corleone
        Aqua.test_all(Corleone)
    end
    @safetestset "Controls" begin
        include("controls.jl")
    end
    @safetestset "InitialCondition" begin
        include("initializers.jl")
    end
    @safetestset "Single shooting" begin
        include("single_shooting.jl")
    end
    @safetestset "Parallel shooting" begin
        include("parallel_shooting.jl")
    end
    @safetestset "Multiple shooting" begin
        include("multiple_shooting.jl")
    end
    @testset "Examples" begin
        @safetestset "Lotka Optimal Control" begin
            include("examples/lotka_oc.jl")
        end
        @safetestset "MTK Example" begin
            include("examples/mtk.jl")
        end
    end
    @safetestset "MTK Symbolic Indexing" begin
        include("mtk_symbolic_index.jl")
    end
end
