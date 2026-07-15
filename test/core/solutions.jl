using Test
using Corleone
using Corleone: Solutions
using OrdinaryDiffEqTsit5
using SymbolicIndexingInterface
using SciMLBase: ParameterIndexingProxy

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

# ---------------------------------------------------------------------------
# Trajectory accessors
# ---------------------------------------------------------------------------

@testset "Trajectory accessors" begin
    prob = LotkaVolterra.generate()
    sys = ControlSymbolCache(prob, [:u1, :u2], [:L])
    sol1 = solve(prob, Tsit5(), saveat=0.5, p=[2., 1., 3., 4., 1.0, 0.0], tspan=(0., 5.))
    sol2 = solve(prob, Tsit5(), saveat=0.5, p=[2., 1., 2., 4., 0.0, 1.0], tspan=(5., 12.))
    seg1 = Corleone.Solutions.ShootingSegment((Solutions.ControlSegment(sol1, sys),), sys)
    seg2 = Corleone.Solutions.ShootingSegment((Solutions.ControlSegment(sol2, sys),), sys)
    traj = Trajectory((seg1, seg2), sys)

    @test symbolic_container(traj) === sys

    @test traj.u  == state_values(traj)
    @test traj.u_minimal == Solutions.minimal_state_values(traj)
    @test traj.c  == Solutions.control_values(traj)
    @test traj.t  == current_time(traj)
    @test traj.p  == parameter_values(traj)
    @test traj.ps isa ParameterIndexingProxy

    # Int getindex
    @test traj[1] == traj.u[1]
    @test traj[2] == traj.u[2]

    # Symbolic getindex – known variable
    x_vals = traj[:x]
    @test length(x_vals) == length(traj.t)
    @test x_vals ≈ getindex.(state_values(traj), variable_index(sys, :x))

    # Control variable (:u1) is augmented into state_values via ControlSymbolCache
    u1_vals = traj[:u1]
    @test length(u1_vals) == length(traj.t)

    # Matrix shape: states × timepoints
    M = Matrix(traj)
    @test size(M, 2) == length(traj.t)
end

# ---------------------------------------------------------------------------
# ControlSymbolCache – SII passthroughs
# ---------------------------------------------------------------------------

@testset "ControlSymbolCache SII passthroughs" begin
    prob = LotkaVolterra.generate()
    cache = ControlSymbolCache(prob, [:u1, :u2], [:L])

    # independent variable
    @test is_independent_variable(cache, :t)
    @test independent_variable_symbols(cache) == independent_variable_symbols(prob.f)

    # is_variable: control sym, real state var, parameter
    @test is_variable(cache, :u1)   # control → registered in cache.controls
    @test is_variable(cache, :x)    # real state var
    @test !is_variable(cache, :α)   # parameter

    # variable_index: control branch vs plain-var branch
    ui = variable_index(cache, :u1)
    @test ui isa Int
    xi = variable_index(cache, :x)
    @test xi isa Int
    @test xi != ui

    # variable_symbols includes controls
    vsyms = variable_symbols(cache)
    @test :u1 ∈ vsyms && :u2 ∈ vsyms && :x ∈ vsyms

    # all_variable_symbols = union of sys vars + control keys
    all_vs = all_variable_symbols(cache)
    @test :u1 ∈ all_vs && :x ∈ all_vs

    # all_symbols
    @test all_symbols(cache) == all_symbols(cache.sys)

    # default_values
    @test default_values(cache) == default_values(cache.sys)

    # constant_structure
    @test constant_structure(cache) == constant_structure(cache.sys)

    # is_time_dependent
    @test is_time_dependent(cache)

    # is_parameter: param vs control
    @test is_parameter(cache, :α)
    @test !is_parameter(cache, :u1)   # control is NOT a parameter

    # parameter_index: None for control, index for param
    @test parameter_index(cache, :u1) === nothing
    @test parameter_index(cache, :α) isa Int

    # parameter_symbols excludes controls
    psyms = parameter_symbols(cache)
    @test :u1 ∉ psyms && :u2 ∉ psyms
    @test :α ∈ psyms

    # is_timeseries_parameter
    @test !is_timeseries_parameter(cache, :u1)
    @test !is_timeseries_parameter(cache, :α)
end
