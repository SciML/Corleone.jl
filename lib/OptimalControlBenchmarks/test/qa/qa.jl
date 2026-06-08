using OptimalControlBenchmarks
using Aqua

@testset "Aqua" begin
    Aqua.test_all(
        OptimalControlBenchmarks
    )
end
