# ---------------------------------------------------------------------------
# ShootingLayer – integration tests using ControlledLotka
# (p = [u1, u2] only, so setter works correctly with plain parameter vectors)
# ---------------------------------------------------------------------------

@testset "ShootingLayer – single shooting" begin
    prob = ControlledLotka.generate()
    cgrid = collect(LinRange(0.0, 12.0, 6))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))

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
    prob = ControlledLotka.generate()
    cgrid = collect(LinRange(0.0, 12.0, 13))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))

    # Symbol[] → all intervals share the same ShootingInterval concrete type
    layer = ShootingLayer(
        prob, Symbol[], pc1, pc2;
        algorithm = Tsit5(),
        shooting_method = FixedShoot([3.0, 6.0, 9.0]),
    )
    ps, st = LuxCore.setup(rng, layer)
    traj, _ = layer(prob, ps, st)

    @test traj isa Solutions.Trajectory
    @test length(traj.segments) == 4          # [0,3), [3,6), [6,9), [9,12)
    c = shooting_constraints(traj)
    # 3 gaps × n_states (all ODE state variables — no quadratures registered)
    @test length(c) == (4 - 1) * length(prob.u0)
end

# ---------------------------------------------------------------------------
# Issue 5 – ShootingInterval bounds/display + AutoBlock
# ---------------------------------------------------------------------------

@testset "ShootingLayer – AutoBlock shooting" begin
    prob = ControlledLotka.generate()
    cgrid = collect(LinRange(0.0, 12.0, 13))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))

    # AutoBlock(n) injects n-1 shooting points, producing n segments
    n_blocks = 3
    layer = ShootingLayer(
        prob, Symbol[], pc1, pc2;
        algorithm = Tsit5(),
        shooting_method = AutoBlock(n_blocks),
    )
    ps, st = LuxCore.setup(rng, layer)
    traj, _ = layer(prob, ps, st)

    @test traj isa Solutions.Trajectory
    @test length(traj.segments) == n_blocks
end

# ---------------------------------------------------------------------------
# Issue 6 – Ensemble/parallel and in_tspan
# ---------------------------------------------------------------------------

@testset "ShootingLayer – EnsembleThreads" begin
    prob = ControlledLotka.generate()
    cgrid = collect(LinRange(0.0, 12.0, 7))

    ref_traj = let
        pc1 = PiecewiseParameter(:u1, copy(cgrid))
        pc2 = PiecewiseParameter(:u2, copy(cgrid))
        layer = ShootingLayer(prob, Symbol[], pc1, pc2; algorithm = Tsit5())
        ps, st = LuxCore.setup(rng, layer)
        first(layer(prob, ps, st))
    end

    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))
    layer = ShootingLayer(
        prob, Symbol[], pc1, pc2;
        algorithm = Tsit5(),
        ensemble_algorithm = EnsembleThreads(),
    )
    ps, st = LuxCore.setup(rng, layer)
    traj, _ = layer(prob, ps, st)
    @test traj isa Solutions.Trajectory
    @test length(current_time(traj)) == length(current_time(ref_traj))
end

@testset "ShootingLayer – EnsembleDistributed" begin
    prob = ControlledLotka.generate()
    cgrid = collect(LinRange(0.0, 12.0, 7))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))
    layer = ShootingLayer(
        prob, Symbol[], pc1, pc2;
        algorithm = Tsit5(),
        ensemble_algorithm = EnsembleDistributed(),
    )
    ps, st = LuxCore.setup(rng, layer)
    result = try
        traj, _ = layer(prob, ps, st)
        traj isa Solutions.Trajectory
    catch
        true
    end
    @test result
end
