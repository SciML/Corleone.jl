using OptimalControlBenchmarks
using Test
using SafeTestsets

const GROUP = get(ENV, "CORLEONE_TEST_GROUP", "All")

if GROUP == "All" || GROUP == "Core"
    @test 1 == 1
end

if GROUP == "All" || GROUP == "QA"
    @safetestset "Code quality (Aqua.jl)" begin
        using Aqua
        using OptimalControlBenchmarks
        Aqua.test_all(OptimalControlBenchmarks)
    end
end
