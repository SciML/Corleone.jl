using Test
using JET
using SafeTestsets

@testset "Corleone.jl" begin
    @safetestset "Code quality (Aqua.jl)" begin
        using Aqua
        using Corleone
        Aqua.test_all(Corleone)
    end
    #@safetestset "Controls" begin
    #    include("controls.jl")
    #end
    #@safetestset "InitialCondition" begin
    #    include("initializers.jl")
    #end
    #@safetestset "Multiple shooting" begin
    #    include("multiple_shooting.jl")
    #end
    #@testset "Examples" begin
    #    @safetestset "Lotka" begin
    #        include("examples/lotka_oc.jl")
    #    end
    #    @safetestset "Lotka MS" begin
    #        include("examples/lotka_ms.jl")
    #    end
    #    @safetestset "Lotka MTK" begin
    #        include("examples/mtk.jl")
    #    end
    #end
end

# What to test?
# local_controls.jl:
#   - construction of index_grid, get_subvector_indices -> Julius
# general:
#   - more convergence? Lotka OED
