using CorleoneCore
using Test
using LuxCore
using Random
using SciMLBase

rng = Random.seed!()

@testset "Global" begin 
    foop_global(p) = [
        p[1]^2, 
        sum(p[2:5]), 
        prod(p)
    ]

    f_g_oop = GridFunction{false}(
        foop_global
    )

    @test !SciMLBase.isinplace(f_g_oop)
    @test CorleoneCore.contains_timegrid_layer(f_g_oop)
    @test !CorleoneCore.contains_tstop_layer(f_g_oop)
    @test !CorleoneCore.contains_saveat_layer(f_g_oop)
    
    f_g_iip = GridFunction{true}(
        (res, args...) -> res .= foop_global(args...)
    )

    @test SciMLBase.isinplace(f_g_iip)
    @test CorleoneCore.contains_timegrid_layer(f_g_iip)
    @test !CorleoneCore.contains_tstop_layer(f_g_iip)
    @test !CorleoneCore.contains_saveat_layer(f_g_iip)

    x0 = randn(rng, 100)
    res = zeros(3)

    ps, st = LuxCore.setup(rng, f_g_iip)
    @inferred f_g_iip((res, x0), ps, st)
    @test all(f_g_iip((res, x0), ps, st) .== (foop_global(x0), st))

    ps, st = LuxCore.setup(rng, f_g_oop)
    @inferred f_g_oop(x0, ps, st)
    @test all(f_g_oop(x0, ps, st) .== (foop_global(x0), st))
    @test res == first(f_g_oop(x0, ps, st))
end

@testset "Time dependent" begin 
    foop_global(x, p, t) = [
        sum(x.^3), sum(p), sin(t) 
    ]

    t = [0., 10., 20.0]
    tspan = (0., 20.0)

    f_g_oop = GridFunction{false}(
        foop_global, t
    )

    @test !SciMLBase.isinplace(f_g_oop)
    @test CorleoneCore.contains_timegrid_layer(f_g_oop)
    @test !CorleoneCore.contains_tstop_layer(f_g_oop)
    @test CorleoneCore.contains_saveat_layer(f_g_oop)
    
    f_g_iip = GridFunction{true}(
        (res, args...) -> res .= foop_global(args...),  t
    )
        
    @test SciMLBase.isinplace(f_g_iip)
    @test CorleoneCore.contains_timegrid_layer(f_g_iip)
    @test !CorleoneCore.contains_tstop_layer(f_g_iip)
    @test CorleoneCore.contains_saveat_layer(f_g_iip)
    
    x0 = Tuple((randn(rng, 2), randn(rng, 3), ti) for ti in t)
    x_iip = Tuple(((zeros(3), xi...) for xi in x0))
    baseline = map(x->foop_global(x...), x0)

    ps, st = LuxCore.setup(rng, f_g_iip)
    @test collect_saveat(st, tspan) == t 
    @test isempty(collect_tstops(st, tspan)) 
    @inferred f_g_iip(x_iip, ps, st)
    @test all(first.(first(f_g_iip(x_iip, ps, st))) .== baseline)

    ps, st = LuxCore.setup(rng, f_g_oop)

    @test collect_saveat(st, tspan) == t 
    @test isempty(collect_tstops(st, tspan)) 
    @inferred f_g_oop(x0, ps, st)
    @test f_g_oop(x0, ps, st) == (reduce(vcat, baseline), st)
    @test reduce(vcat, first.(x_iip)) == first(f_g_oop(x0, ps, st))
end