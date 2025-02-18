using CorleoneCore
using SciMLBase
using OrdinaryDiffEqTsit5
using LuxCore
using Random
using Test

@testset "Basic Problem" begin 
    ## Define a simple problem
    function dynamics(du, u, p, t)
        du .= -p .* u 
    end


    problem = ODEProblem(dynamics, randn(2), (0., 1.), rand(2))
    solve(problem, Tsit5(), saveat = 0.1)

    # Define the grid 
    grid = SimulationGrid(
        (0., 1.) => (; u0 = ones(2)), 
        (1., 2.) => (; u0 = ones(2)),
        solver = Tsit5()
    )

    ps, st = LuxCore.setup(Random.seed!(), grid)
    @test_nowarn grid(problem, ps, st)

    sols, st = grid(problem, ps, st)    
    @test length(sols) == 2
    @test CorleoneCore.contains_timegrid_layer(grid)
    @test !CorleoneCore.contains_tstop_layer(grid)
    @test !CorleoneCore.contains_saveat_layer(grid)
    @test isempty(CorleoneCore.collect_saveat(st, (0., 10.)))
    @test isempty(CorleoneCore.collect_tstops(st, (0., 10.)))
end