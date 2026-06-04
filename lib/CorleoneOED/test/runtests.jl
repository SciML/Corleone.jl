using CorleoneOED
using SafeTestsets

const GROUP = get(ENV, "CORLEONE_TEST_GROUP", "All")

if GROUP == "All" || GROUP == "Core"
    @safetestset "1D Example" begin
        include("1d_oed.jl")
    end
    @safetestset "Lotka Volterra" begin
        include("lotka_oed.jl")
    end
    @safetestset "Lotka Volterra SVD" begin
        include("lotka_oed_svd.jl")
    end
end

if GROUP == "All" || GROUP == "QA"
    @safetestset "Code quality (Aqua.jl)" begin
        using Aqua
        using CorleoneOED
        Aqua.test_all(CorleoneOED)
    end
end
