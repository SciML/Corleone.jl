using Test
using Corleone
using Corleone: Solutions
using OrdinaryDiffEqTsit5
using StableRNGs
using LuxCore
using SymbolicIndexingInterface
using SciMLBase

include(joinpath(@__FILE__, "..", "..", "helper.jl"))

rng = StableRNG(42)

# ---------------------------------------------------------------------------
# PiecewiseParameter
# ---------------------------------------------------------------------------

@testset "PiecewiseParameter" begin
    @testset "Construction" begin
        pc = PiecewiseParameter(:u, [0.0, 1.0, 2.0])
        @test pc.parameter_id == :u
        @test pc.tpoints == [0.0, 1.0, 2.0]
        @test isnothing(pc.bounds)
        @test isnothing(pc.init)
        @test isempty(pc.injected)
        @test LuxCore.parameterlength(pc) == 4   # 3 breakpoints → 4 intervals
    end

    @testset "initialparameters – zeros by default" begin
        pc = PiecewiseParameter(:u, [0.0, 1.0, 2.0])
        ps, _ = LuxCore.setup(rng, pc)
        @test length(ps) == 4
        @test all(all(iszero, p) for p in ps)
    end

    @testset "initialstates" begin
        pc = PiecewiseParameter(:u, [0.0, 1.0, 2.0])
        _, st = LuxCore.setup(rng, pc)
        @test st.tpoints == pc.tpoints
        @test st.first_index == firstindex(pc.tpoints)
        @test st.last_index == lastindex(pc.tpoints)
    end

    @testset "Evaluation – correct interval selected" begin
        pc = PiecewiseParameter(:u, [0.0, 1.0, 2.0])
        _, st = LuxCore.setup(rng, pc)
        ps = [fill(Float64(i), 1) for i in 1:4]
        # t < 0.0 → interval 1
        v1, _ = pc(-0.1, ps, st)
        @test v1 == ps[1]
        # 0.0 ≤ t < 1.0 → interval 2
        v2, _ = pc(0.5, ps, st)
        @test v2 == ps[2]
        # 1.0 ≤ t < 2.0 → interval 3
        v3, _ = pc(1.0, ps, st)
        @test v3 == ps[3]
        # t ≥ 2.0 → interval 4 (last)
        v4, _ = pc(3.0, ps, st)
        @test v4 == ps[4]
    end

    @testset "inject! inserts breakpoint and updates injected" begin
        pc = PiecewiseParameter(:u, [0.0, 1.0, 2.0])
        inject!(pc, 0.5)
        @test 0.5 ∈ pc.tpoints
        @test issorted(pc.tpoints)
        @test !isempty(pc.injected)
    end

    @testset "reset! removes injected breakpoints" begin
        pc = PiecewiseParameter(:u, [0.0, 1.0, 2.0])
        original = copy(pc.tpoints)
        inject!(pc, 0.5)
        reset!(pc)
        @test pc.tpoints == original
        @test isempty(pc.injected)
    end

    @testset "shooting_constraints at injection points" begin
        pc = PiecewiseParameter(:u, [0.0, 2.0])
        inject!(pc, 1.0)
        ps = [fill(1.0, 1), fill(3.0, 1), fill(5.0, 1)]
        c = shooting_constraints(pc, ps, nothing)
        expected = ps[pc.injected] .- ps[pc.injected .- 1]
        @test c == expected
    end
end

# ---------------------------------------------------------------------------
# ShootingInterval
# ---------------------------------------------------------------------------

@testset "ShootingInterval" begin
    prob = LotkaVolterra.generate()

    @testset "Construction with numeric variable_id" begin
        si = ShootingInterval(prob, [1, 2], prob.tspan)
        @test si.tspan == prob.tspan
        @test isnothing(si.bounds)
    end

    @testset "Construction with symbolic variable_id" begin
        si = ShootingInterval(prob, [:x, :y], (0.0, 5.0))
        @test si.tspan == (0.0, 5.0)
    end

    @testset "initialparameters seeds from prob.u0" begin
        si = ShootingInterval(prob, [1, 2], prob.tspan)
        ps_si, _ = LuxCore.setup(rng, si)
        @test ps_si ≈ prob.u0[1:2]
    end

    @testset "Calling ShootingInterval remakes problem" begin
        si = ShootingInterval(prob, [1, 2], (0.0, 5.0))
        _, st_si = LuxCore.setup(rng, si)
        new_u0 = [0.9, 0.8]
        new_prob, _ = si(prob, new_u0, st_si)
        @test new_prob.u0[1:2] ≈ new_u0
        @test new_prob.u0[3] == prob.u0[3]   # non-free component unchanged
        @test new_prob.tspan == (0.0, 5.0)
    end
end

# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

@testset "Controls" begin
    prob = LotkaVolterra.generate()
    sys  = symbolic_container(prob.f)

    # 5-breakpoint grid spanning tspan  DimensionMismatch: new dimensions (6,) must be consistent with array length 2
    cgrid = collect(LinRange(0.0, 12.0, 6))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))

    @testset "Construction" begin
        c = Controls(pc1, pc2; sys = sys)
        reset!(c)
        @test length(c.controls) == 2
        @test length(c.permutation) == 2
    end

    @testset "collect_timegrid covers tspan" begin
        c = Controls(pc1, pc2; sys = sys)
        reset!(c)
        _, st = LuxCore.setup(rng, c)
        tspans = Corleone.collect_timegrid(c, nothing, st, (0.0, 12.0))
        @test first(first(tspans)) == 0.0
        @test last(last(tspans)) == 12.0
    end

    @testset "Evaluation returns vector matching ncontrols" begin
        c = Controls(pc1, pc2; sys = sys)
        reset!(c)
        ps, st = LuxCore.setup(rng, c)
        result, _ = c(0.5, ps, st)
        @test length(result) == 2
    end
end

# ---------------------------------------------------------------------------
# ShootingLayer – integration tests using ControlledLotka
# (p = [u1, u2] only, so setter works correctly with plain parameter vectors)
# ---------------------------------------------------------------------------

@testset "ShootingLayer – single shooting" begin
    prob  = ControlledLotka.generate()
    cgrid = collect(LinRange(0.0, 12.0, 6))
    pc1   = PiecewiseParameter(:u1, copy(cgrid))
    pc2   = PiecewiseParameter(:u2, copy(cgrid))

    # Symbol[] → no tunable ICs; fixed initial condition from the problem
    layer = ShootingLayer(prob, Symbol[], pc1, pc2; algorithm = Tsit5())
    ps, st = LuxCore.setup(rng, layer)
    traj, _ = layer(prob, ps, st)

    @test traj isa Solutions.Trajectory
    @test length(traj.segments) == 1
    @test length(current_time(traj)) > 0
    # single shooting → no continuity constraints
    @test isempty(shooting_constraints(traj))
end

@testset "ShootingLayer – multiple shooting (FixedShoot)" begin
    prob  = ControlledLotka.generate()
    cgrid = collect(LinRange(0.0, 12.0, 13))
    pc1   = PiecewiseParameter(:u1, copy(cgrid))
    pc2   = PiecewiseParameter(:u2, copy(cgrid))

    # Symbol[] → all intervals share the same ShootingInterval concrete type
    layer = ShootingLayer(
        prob, Symbol[], pc1, pc2;
        algorithm        = Tsit5(),
        shooting_method  = FixedShoot([3.0, 6.0, 9.0]),
    )
    ps, st = LuxCore.setup(rng, layer)
    traj, _ = layer(prob, ps, st)

    @test traj isa Solutions.Trajectory
    @test length(traj.segments) == 4          # [0,3), [3,6), [6,9), [9,12)
    c = shooting_constraints(traj)
    # 3 gaps × n_states (all ODE state variables — no quadratures registered)
    @test length(c) == (4 - 1) * length(prob.u0)
end
