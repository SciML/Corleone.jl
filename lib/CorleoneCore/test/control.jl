using Pkg
Pkg.activate(joinpath(pwd(), "lib/CorleoneCore"))

using CorleoneCore
using CorleoneCore: get_bounds, get_timegrid, get_controls, check_consistency
using Test
using Random

rng = Random.seed!(42)

@testset "Default" begin 
    c1 = ControlParameter(0.0:0.1:1.0)
    @test 0.0:0.1:1.0 ==  get_timegrid(c1)
    @test issorted(get_timegrid(c1))
    u = get_controls(rng, c1)
    @test length(u) == length(get_timegrid(c1))
    @test length(u) == length(first(get_bounds(c1)))
    @test length(u) == length(last(get_bounds(c1)))
    @test all(first(get_bounds(c1)) .== -Inf)
    @test all(last(get_bounds(c1)) .== Inf)
    @test_nowarn check_consistency(rng, c1)
end

@testset "Custom Bounds" begin 
    c1 = ControlParameter(0.0:0.1:1.0, bounds = (t)->(zero(t), zero(t) .+ 1))
    @test 0.0:0.1:1.0 ==  get_timegrid(c1)
    @test issorted(get_timegrid(c1))
    u = get_controls(rng, c1)
    @test length(u) == length(get_timegrid(c1))
    @test length(u) == length(first(get_bounds(c1)))
    @test length(u) == length(last(get_bounds(c1)))
    @test all(first(get_bounds(c1)) .== 0.)
    @test all(last(get_bounds(c1)) .== 1.)
    @test_nowarn check_consistency(rng, c1)
end

@testset "Custom Controls" begin 
    c1 = ControlParameter(0.0:0.1:1.0, controls = (rng, t, bounds)->sin.(t))
    @test 0.0:0.1:1.0 ==  get_timegrid(c1)
    @test issorted(get_timegrid(c1))
    u = get_controls(rng, c1)
    @test length(u) == length(get_timegrid(c1))
    @test u == sin.(0.0:0.1:1.0)
    @test_nowarn check_consistency(rng, c1)
    c1 = ControlParameter(0.0:0.1:1.0, controls = (rng, t, bounds)->sin.(t), bounds = (t)->(zero(t), zero(t) .+ 0.1))
    @test_throws AssertionError check_consistency(rng, c1)
end

@testset "Utility" begin 
c1 = ControlParameter(0.0:0.1:1.0)
c2 = ControlParameter(0.5:0.5:5.0)
c3 = ControlParameter(0.25:0.5:1.5)
controls = (c1, c2, c3)
# Check timespans
ts = map(get_timegrid, controls)
lengths = sum(length, ts)
tspans = CorleoneCore.collect_tspans(c1, c2, c3)
tgrid = reduce(vcat, ts) |> unique! |> sort! 
@test all(map(ts) do ti 
    # Check if all individual grids are present
    all(∈(tgrid), ti)
end)
idx = CorleoneCore.build_index_grid(c1, c2, c3)
@test (1,lengths) == extrema(idx)
@test size(idx) == (3, length(tgrid))
u0 = CorleoneCore.collect_local_controls(rng, c1, c2, c3)
@test size(u0) == (lengths,)
end