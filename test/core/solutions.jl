using Test
using Corleone
using Corleone: Solutions
using OrdinaryDiffEqTsit5
using SymbolicIndexingInterface

include(joinpath(@__FILE__, "..", "..", "helper.jl"))

@testset "ControlCache" begin
    @testset "Predefined system" begin
        prob = LotkaVolterra.generate()
        cache_num = ControlSymbolCache(prob, [5, 6], [3])
        cache_mixed = ControlSymbolCache(prob, [:u1, :u2], [3])
        cache_sym = ControlSymbolCache(prob, [:u1, :u2], [:L])
        @test variable_index(cache_num, :u1) == 4
        @test variable_index(cache_num, :u2) == 5
        @test parameter_index(cache_num, :u1) == nothing
        @test parameter_index(cache_num, :u2) == nothing
        @test independent_variable_symbols(prob.f) == independent_variable_symbols(cache_num)
        @test length(cache_mixed.controls) == 2
        @test length(cache_mixed.quadratures) == 1
        @test cache_num.controls == cache_sym.controls == cache_mixed.controls
        @test cache_num.quadratures == cache_sym.quadratures == cache_mixed.quadratures
    end
    @testset "Fallback system" begin
        prob = LotkaVolterra.generate(; sys=nothing)
        cache_num = ControlSymbolCache(prob, [5, 6], [3])
        @test_throws AssertionError ControlSymbolCache(prob, [:u1, :u2], [3])
        @test_throws AssertionError ControlSymbolCache(prob, [:u1, :u2], [:L])
        cache_mixed = ControlSymbolCache(prob, [:p₅, :p₆], [3])
        cache_sym = ControlSymbolCache(prob, [:p₅, 6], [:u₃])
        @test variable_index(cache_num, :p₅) == 4
        @test variable_index(cache_num, :p₆) == 5
        @test parameter_index(cache_num, :p₅) == nothing
        @test parameter_index(cache_num, :p₆) == nothing
        @test [:t] == independent_variable_symbols(cache_num)
        @test length(cache_mixed.controls) == 2
        @test length(cache_mixed.quadratures) == 1
        @test cache_num.controls == cache_sym.controls == cache_mixed.controls
        @test cache_num.quadratures == cache_sym.quadratures == cache_mixed.quadratures
    end
end

@testset "ControlSegment" begin
    prob = LotkaVolterra.generate()
    sol = solve(prob, Tsit5(), p=[2., 1., 3., 4., 0.0, 5.0])
    sys = ControlSymbolCache(prob, [:u1, :u2], [:L])
    seg = Corleone.Solutions.ControlSegment(sol, sys)
    @test symbolic_container(seg) == sys
    @test state_values(seg) == vcat.(sol.u, fill([0.0, 5.0], length(sol.t)))
    @test Solutions.minimal_state_values(seg) == map(Base.Fix2(getindex, 1:2), sol.u)
    @test Solutions.control_values(seg) == [0.0, 5.0]
    @test parameter_values(seg) == [2., 1., 3., 4.]
    @test current_time(seg) == current_time(sol)
end

@testset "ShootingSegment" begin
    prob = LotkaVolterra.generate()
    sol1 = solve(prob, Tsit5(), p=[2., 1., 3., 4., 0.0, 5.0], tspan=(0., 5.))
    sol2 = solve(sol1.prob, Tsit5(), u0=sol1.u[end], tspan=(5., 12.))
    sys = ControlSymbolCache(prob, [:u1, :u2], [:L])
    seg_cont = Corleone.Solutions.ShootingSegment(map(Base.Fix2(Solutions.ControlSegment, sys), (sol1, sol2)), sys)
    @test length(current_time(seg_cont)) == length(current_time(sol1)) + length(current_time(sol2)) - 1
    @test minimal_state_values(seg_cont) == vcat(map(Base.Fix2(getindex, 1:2), sol1.u)[1:(end-1)], map(Base.Fix2(getindex, 1:2), sol2.u))
    @test parameter_values(seg_cont) == [2., 1., 3., 4.]
    @test Solutions.control_values(seg_cont) == [[0., 5.], [0., 5.]]
    @test state_values(seg_cont) == vcat(vcat.(sol1.u[1:(end-1)], fill([0.0, 5.0], length(sol1.u)-1)), vcat.(sol2.u, fill([0.0, 5.0], length(sol2.u))))
end

@testset "Trajectory" begin
    prob = LotkaVolterra.generate()
    sys = ControlSymbolCache(prob, [:u1, :u2], [:L])
    sol1 = solve(prob, Tsit5(), saveat=0.2, p=[2., 1., 3., 4., 1.0, 0.0], tspan=(0., 5.))
    sol2 = solve(prob, Tsit5(), saveat=0.2, p=[2., 1., 2., 4., 0.0, 1.0], tspan=(5., 12.))
    seg1 = Corleone.Solutions.ShootingSegment((Solutions.ControlSegment(sol1, sys),), sys)
    seg2 = Corleone.Solutions.ShootingSegment((Solutions.ControlSegment(sol2, sys),), sys)
    traj = Trajectory((seg1, seg2), sys)
    @test current_time(traj) == vcat(sol1.t[1:(end-1)], sol2.t)
    @test minimal_state_values(traj) == vcat(minimal_state_values(seg1)[1:(end-1)], minimal_state_values(seg2))
    @test control_values(traj) == vcat(control_values(seg1), control_values(seg2))
    @test state_values(traj) == vcat(
        vcat.(sol1.u[1:(end-1)], fill([1.0, 0.0], length(sol1.t)-1)),
        vcat.(map(Base.Fix1(+, [0., 0., sol1.u[end][3]]), sol2.u), fill([0.0, 1.0], length(sol2.t)))
    )
end



