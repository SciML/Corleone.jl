# ---------------------------------------------------------------------------
# abstract.jl – bound helpers and constraint machinery
# ---------------------------------------------------------------------------

@testset "abstract.jl – get_lower/upper_bound for Number, Array, Tuple" begin
    # Number branch (lines 49-50)
    @test Corleone.get_lower_bound(1.0f0) === -Inf32
    @test Corleone.get_upper_bound(1.0f0) === Inf32
    @test Corleone.get_lower_bound(2.0) === -Inf
    @test Corleone.get_upper_bound(2.0) === Inf

    # AbstractArray branch (line 53-54)
    lb_arr = Corleone.get_lower_bound([1.0, 2.0])
    @test lb_arr == [-Inf, -Inf]
    ub_arr = Corleone.get_upper_bound([1.0, 2.0])
    @test ub_arr == [Inf, Inf]

    # AbstractVecOrTuple branch (tuple)
    lb_tup = Corleone.get_lower_bound((1.0, 2.0))
    @test lb_tup == (-Inf, -Inf)
    ub_tup = Corleone.get_upper_bound((1.0, 2.0))
    @test ub_tup == (Inf, Inf)

    # NamedTuple branch
    lb_nt = Corleone.get_lower_bound((; a = 1.0, b = 2.0f0))
    @test lb_nt.a === -Inf && lb_nt.b === -Inf32
end

@testset "abstract.jl – get_lower/upper_bound and get_bounds on layer" begin
    prob = LotkaVolterra.generate()
    sys = symbolic_container(prob.f)
    cgrid = collect(LinRange(0.0, 12.0, 4))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))
    c = Controls(pc1, pc2; sys = sys)
    reset!(c)
    ps, st = LuxCore.setup(rng, c)

    # Container layer: get_lower/upper_bound returns a NamedTuple keyed by T=(:controls,),
    # matching the same shape as ps/st themselves (via the shared traverse_leaves combinator).
    lb_c = Corleone.get_lower_bound(c, ps, st)
    ub_c = Corleone.get_upper_bound(c, ps, st)
    @test lb_c isa NamedTuple
    @test ub_c isa NamedTuple
    @test haskey(lb_c, :controls) && haskey(ub_c, :controls)
    # get_bounds returns (lb, ub) tuple
    bounds = Corleone.get_bounds(c, ps, st)
    @test bounds == (lb_c, ub_c)
end

@testset "abstract.jl – shooting_constraints on PiecewiseParameter (3-arg)" begin
    pc = PiecewiseParameter(:u, [0.0, 1.0])
    ps, st = LuxCore.setup(rng, pc)
    # PiecewiseParameter has a 3-arg method; with no injected points returns empty
    c = shooting_constraints(pc, ps, st)
    @test isempty(c)
end

@testset "abstract.jl – shooting_constraints on a Controls container" begin
    # Regression coverage: container-level shooting_constraints was previously
    # untested end-to-end (only the PiecewiseParameter leaf method above was).
    prob = LotkaVolterra.generate()
    sys = symbolic_container(prob.f)
    pc1 = PiecewiseParameter(:u1, [0.0, 2.0])
    pc2 = PiecewiseParameter(:u2, [0.0, 2.0])
    c = Controls(pc1, pc2; sys = sys)
    reset!(c)

    # No injected shooting points on either control → no constraints.
    ps, st = LuxCore.setup(rng, c)
    @test isempty(shooting_constraints(c, ps, st))

    # Inject a shooting point into u1 only; u2 stays untouched.
    inject!(pc1, 1.0)
    ps, st = LuxCore.setup(rng, c)
    expected = shooting_constraints(pc1, ps.controls.u1, st.controls.u1)
    @test shooting_constraints(c, ps, st) == expected
end

@testset "abstract.jl – get_number_of_shooting_constraints recursion" begin
    # Leaf: AbstractLuxLayer default is 0; PiecewiseParameter overrides it.
    @test Corleone.get_number_of_shooting_constraints(PiecewiseParameter(:u, [0.0, 1.0])) == 0

    pc1 = PiecewiseParameter(:u1, [0.0, 2.0])
    pc2 = PiecewiseParameter(:u2, [0.0, 2.0])

    # ContainerLayer over a NamedTuple of leaves (Controls.controls): sums children.
    prob = LotkaVolterra.generate()
    sys = symbolic_container(prob.f)
    c = Controls(pc1, pc2; sys = sys)
    inject!(pc1, 1.0)
    inject!(pc2, 0.5)
    inject!(pc2, 1.5)
    @test Corleone.get_number_of_shooting_constraints(c) == 1 + 2

    # ShootingLayer overrides the generic container recursion: what it actually
    # contributes once solved is state-continuity constraints (one per
    # non-quadrature state per gap between intervals, per
    # Solutions.Trajectory.shooting_constraints), independent of any
    # PiecewiseParameter-injected breakpoints on its controls.
    single = ShootingLayer(
        ControlledLotka.generate(), Symbol[], pc1, pc2;
        algorithm = Tsit5(),
    )
    @test Corleone.get_number_of_shooting_constraints(single) == 0
    inject!(single.controls.controls.u1, 1.0)
    @test Corleone.get_number_of_shooting_constraints(single) == 0

    multi_prob = ControlledLotka.generate()
    cgrid = collect(LinRange(0.0, 12.0, 13))
    pc3 = PiecewiseParameter(:u1, copy(cgrid))
    pc4 = PiecewiseParameter(:u2, copy(cgrid))
    multi = ShootingLayer(
        multi_prob, Symbol[], pc3, pc4;
        algorithm = Tsit5(), shooting_method = FixedShoot([3.0, 6.0, 9.0]),
    )
    ps, st = LuxCore.setup(rng, multi)
    traj, _ = multi(multi_prob, ps, st)
    @test Corleone.get_number_of_shooting_constraints(multi) == length(Solutions.shooting_constraints(traj))
end

@testset "abstract.jl – collect_activity_pattern layer variants" begin
    prob = LotkaVolterra.generate()
    sys = symbolic_container(prob.f)
    cgrid = collect(LinRange(0.0, 12.0, 4))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))
    c = Controls(pc1, pc2; sys = sys)
    reset!(c)
    ps, st = LuxCore.setup(rng, c)

    tpoints = [0.0, 4.0, 8.0, 12.0]

    # PiecewiseParameter (AbstractLuxLayer) → sparse matrix
    pat_pc = Corleone.collect_activity_pattern(tpoints, pc1, ps.controls.u1, st.controls.u1)
    @test size(pat_pc, 1) == length(tpoints)

    # Controls (AbstractLuxContainerLayer) → NamedTuple
    pat_c = Corleone.collect_activity_pattern(tpoints, c, ps, st)
    @test pat_c isa NamedTuple
    @test haskey(pat_c, :controls)
end

@testset "Layers.jl bound helpers – maybecallme / first_or_first / last_or_last" begin
    # Callable-tuple bounds exercises maybecallme(f, ps, st) via first_or_first/last_or_last
    f_lb = (ps, st) -> fill(-3.0, length(ps))
    f_ub = (ps, st) -> fill(4.0, length(ps))
    pc = PiecewiseParameter(:u, [0.0, 1.0], nothing, (f_lb, f_ub))
    ps, st = LuxCore.setup(rng, pc)
    lb = Corleone.get_lower_bound(pc, ps, st)
    ub = Corleone.get_upper_bound(pc, ps, st)
    @test all(x -> x == -3.0, lb)
    @test all(x -> x == 4.0, ub)
end

@testset "abstract.jl – container traversal works with ComponentArrays ps/st" begin
    # Regression test: container-level dispatch in abstract.jl must use getproperty
    # (not getfield) so ComponentArrays.ComponentVector, which only overloads
    # getproperty, can stand in for the usual NamedTuple ps.
    using ComponentArrays

    prob = LotkaVolterra.generate()
    sys = symbolic_container(prob.f)
    cgrid = collect(LinRange(0.0, 12.0, 4))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))
    c = Controls(pc1, pc2; sys = sys)
    reset!(c)
    ps, st = LuxCore.setup(rng, c)
    ps_ca = ComponentArray(ps)

    @test Corleone.get_timepoints(c, ps_ca, st) == Corleone.get_timepoints(c, ps, st)
    @test Corleone.get_lower_bound(c, ps_ca, st) == Corleone.get_lower_bound(c, ps, st)
    @test Corleone.get_upper_bound(c, ps_ca, st) == Corleone.get_upper_bound(c, ps, st)
    @test Corleone.collect_activity_pattern([0.0, 4.0, 8.0, 12.0], c, ps_ca, st) isa NamedTuple
end
