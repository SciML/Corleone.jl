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

    # Container layer: get_lower/upper_bound returns nested structure over T=(:controls,)
    lb_c = Corleone.get_lower_bound(c, ps, st)
    ub_c = Corleone.get_upper_bound(c, ps, st)
    @test lb_c isa Tuple   # map over T=(:controls,) returns a 1-tuple
    @test ub_c isa Tuple
    # get_bounds returns (lb, ub) tuple
    bounds = Corleone.get_bounds(c, ps, st)
    @test bounds == (lb_c, ub_c)
end

@testset "abstract.jl – number_of_shooting_constraints (3-arg) on layer" begin
    # Generic AbstractLuxLayer 3-arg always returns 0
    pc = PiecewiseParameter(:u, [0.0, 1.0, 2.0])
    inject!(pc, 0.5)
    ps, st = LuxCore.setup(rng, pc)
    @test number_of_shooting_constraints(pc, ps, st) == 0   # 3-arg generic returns 0

    # Container: Controls with no injected pts → nested NamedTuple result
    prob = LotkaVolterra.generate()
    sys = symbolic_container(prob.f)
    cgrid = collect(LinRange(0.0, 12.0, 4))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))
    c = Controls(pc1, pc2; sys = sys)
    reset!(c)
    ps, st = LuxCore.setup(rng, c)
    n = number_of_shooting_constraints(c, ps, st)
    @test n isa NamedTuple   # nested result from container dispatch
end

@testset "abstract.jl – shooting_constraints on PiecewiseParameter (3-arg)" begin
    pc = PiecewiseParameter(:u, [0.0, 1.0])
    ps, st = LuxCore.setup(rng, pc)
    # PiecewiseParameter has a 3-arg method; with no injected points returns empty
    c = shooting_constraints(pc, ps, st)
    @test isempty(c)
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
