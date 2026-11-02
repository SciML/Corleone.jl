using OptimalControlBenchmarks, Test
const OCB = OptimalControlBenchmarks

@testset "Grid scaling preserves reusable configurations" begin
    grids = BenchmarkGrids(; control_grid = [0.0, 0.5], shooting_grid = [0.0, 0.5, 1.0], constraint_grid = [0.0, 1.0])
    scaled = OCB.scale_grids((2.0, 4.0), grids)
    @test scaled.control_grid == [2.0, 3.0]
    @test scaled.shooting_grid == [2.0, 3.0, 4.0]
    @test scaled.constraint_grid == [2.0, 4.0]
    @test grids.control_grid == [0.0, 0.5]
    @test grids.shooting_grid == [0.0, 0.5, 1.0]
    @test grids.constraint_grid == [0.0, 1.0]
    @test scaled.control_grid !== grids.control_grid
    @test scaled.shooting_grid !== grids.shooting_grid
    @test scaled.constraint_grid !== grids.constraint_grid
    @test OCB.scale_grids!((4.0, 8.0), grids) === grids
    @test grids.control_grid == [4.0, 6.0]
    for invalid in ([0.5, 0.0], [0.0, 0.0], [-0.1, 0.5], [0.0, 1.1], [0.0, NaN])
        data = BenchmarkGrids([0.0, 0.5], invalid, [0.0, 1.0])
        @test_throws ArgumentError OCB.scale_grids!((0.0, 10.0), data)
        @test data.control_grid == [0.0, 0.5] # No partial writes on validation failure.
    end
    @test_throws ArgumentError OCB.scale_grids((1.0, 0.0), BenchmarkGrids(; control_grid = []))
    @test_throws ArgumentError OCB.scale_grids((0.0, Inf), BenchmarkGrids(; control_grid = []))
end

@testset "Registry and all symbolic benchmark definitions" begin
    catalog = load_benchmarks()
    names = nameof.(catalog)
    expected = [Symbol(splitext(f)[1]) for f in readdir(joinpath(@__DIR__, "..", "..", "src", "problems")) if endswith(f, ".jl")]
    @test names == expected
    @test allunique(names)
    @test all(a === b for (a, b) in zip(catalog, load_benchmarks()))
    methods_before = collect(methods(OCB.lqr))
    empty!(catalog)
    @test !isempty(load_benchmarks())
    @test collect(methods(OCB.lqr)) == methods_before
    grids = BenchmarkGrids(; control_grid = [0.0, 0.5], shooting_grid = [0.0, 0.5, 1.0], constraint_grid = [0.0, 0.5, 1.0])
    for benchmark in load_benchmarks()
        data = benchmark(grids)
        @test first(data.tspan) < last(data.tspan)
        @test data.grids.control_grid == first(data.tspan) .+ (last(data.tspan) - first(data.tspan)) .* [0.0, 0.5]
        @test data.grids.shooting_grid == collect(range(data.tspan...; length = 3))
        @test data.dims[1] == length(OCB.ModelingToolkit.equations(data.system))
        @test data.dims[2] == length(OCB.inputs(data.system))
        @test !isempty(data.name)
        @test grids.control_grid == [0.0, 0.5]
        @test grids.shooting_grid == [0.0, 0.5, 1.0]
        @test grids.constraint_grid == [0.0, 0.5, 1.0]
    end
    wrapped = OCB.OptimalControlBenchmark(:wrapped_lqr, "LQR wrapper", OCB.lqr)
    @test wrapped(grids).tspan == OCB.lqr(grids).tspan
end
