using CorleoneCore
using Test
using Random
using LuxCore

rng = Random.seed!(42)

@testset "Discrete Parameter" begin
    p0 = [1., 2., 3.]
    tstops = [0., 1., 2.]
    teval = [-1., 0.2, 1.3, 2.5, 10.0]
    pidx = [1, 1, 2, 3, 3]
    p_discrete = Parameter(p0; tstops)
    @test CorleoneCore.contains_timegrid_layer(p_discrete)
    @test CorleoneCore.contains_tstop_layer(p_discrete)
    @test !CorleoneCore.contains_saveat_layer(p_discrete)
    ps, st = LuxCore.setup(rng, p_discrete)
    @test collect_tstops(st, (0., 2.)) == tstops
    @test isempty(collect_saveat(st, (0., 2.)))
    for (ti, id) in zip(teval, pidx)
        p_t, st_ = p_discrete(ti, ps, st)
        @inferred p_discrete(ti, ps, st)
        @test p_t == selectdim(p0, 1, id)
        @test st_ == st
    end
end

@testset "Constant Parameter" begin
    p0 = [1., 2., 3.]
    teval = [-1., 0.2, 1.3, 2.5, 10.0]
    pidx = [1, 1, 2, 3, 3]
    p_cont = Parameter(p0)
    @test CorleoneCore.contains_timegrid_layer(p_cont)
    @test !CorleoneCore.contains_tstop_layer(p_cont)
    @test !CorleoneCore.contains_saveat_layer(p_cont)
    ps, st = LuxCore.setup(rng, p_cont)
    @test isempty(collect_tstops(st, (0., 2.)))
    @test isempty(collect_saveat(st, (0., 2.)))
    for (ti, id) in zip(teval, pidx)
        p_t, st_ = p_cont(ti, ps, st)
        @inferred p_cont(ti, ps, st)
        @test p_t == selectdim(p0, 1, :)
        @test st_ == st
    end
end

@testset "Bounded Parameter" begin
    p0 = [1., 2., 3.]
    lower_bounds = [0., 1., 2.]
    upper_bounds = [2., 3., 4.]
    @test_nowarn Parameter(p0; lower_bounds, upper_bounds)
    # Fails 
    @test_throws AssertionError Parameter(p0; lower_bounds = upper_bounds, upper_bounds = lower_bounds)
    @test_throws AssertionError Parameter(p0; lower_bounds = lower_bounds[1:2], upper_bounds = upper_bounds)
    @test_throws AssertionError Parameter(p0; lower_bounds = randn(rng, 10), upper_bounds = randn(rng, 10))
end
