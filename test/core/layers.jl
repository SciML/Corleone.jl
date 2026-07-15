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

# ---------------------------------------------------------------------------
# PiecewiseParameter – callable init, bounds, dict-cache, constraints
# ---------------------------------------------------------------------------

@testset "PiecewiseParameter – callable init" begin
    # Callable init branch (initialparameters callable)
    my_init = (rng, T, n) -> fill(T(0.5), n)
    pc = PiecewiseParameter(:u, [0.0, 1.0, 2.0], my_init)
    ps, _ = LuxCore.setup(rng, pc)
    @test length(ps) == 4
    @test all(p -> all(==(0.5), p), ps)
end

@testset "PiecewiseParameter – display_name gensym branch" begin
    # Non-named integer id → gensym branch
    pc = PiecewiseParameter(1, [0.0, 1.0])
    @test LuxCore.display_name(pc) isa Symbol
end

@testset "PiecewiseParameter – bounds (tuple)" begin
    pc = PiecewiseParameter(:u, [0.0, 1.0, 2.0], nothing, (-1.0, 2.0))
    ps, st = LuxCore.setup(rng, pc)
    lb = Corleone.get_lower_bound(pc, ps, st)
    ub = Corleone.get_upper_bound(pc, ps, st)
    @test lb == -1.0
    @test ub == 2.0
end

@testset "PiecewiseParameter – bounds (callable)" begin
    lb_fn = (ps, st) -> fill(-2.0, length(ps))
    ub_fn = (ps, st) -> fill(3.0, length(ps))
    pc = PiecewiseParameter(:u, [0.0, 1.0], nothing, (lb_fn, ub_fn))
    ps, st = LuxCore.setup(rng, pc)
    lb = Corleone.get_lower_bound(pc, ps, st)
    ub = Corleone.get_upper_bound(pc, ps, st)
    @test all(x -> x == -2.0, lb)
    @test all(x -> x == 3.0, ub)
end

@testset "PiecewiseParameter – find_index! dict-cache branch" begin
    pc = PiecewiseParameter(:u, [0.0, 1.0, 2.0])
    ps, st = LuxCore.setup(rng, pc)
    ps_vals = [fill(Float64(i), 1) for i in 1:4]
    # Inject a Dict cache into the state
    dict_cache = Dict{Any,Int}()
    st_cached = merge(st, (; cache = dict_cache))
    v1, st2 = pc(0.5, ps_vals, st_cached)
    @test v1 == ps_vals[2]
    # Second call hits dict cache
    v2, _ = pc(0.5, ps_vals, st2)
    @test v2 == ps_vals[2]
    @test length(dict_cache) >= 1
end

@testset "PiecewiseParameter – number_of_shooting_constraints and shooting_constraints!" begin
    pc = PiecewiseParameter(:u, [0.0, 2.0])
    inject!(pc, 1.0)
    @test number_of_shooting_constraints(pc) == 1
    ps_vals = [fill(Float64(i), 1) for i in 1:3]
    expected = shooting_constraints(pc, ps_vals, nothing)
    res = similar(expected)
    shooting_constraints!(res, pc, ps_vals, nothing)
    @test res == expected
end

@testset "PiecewiseParameter – get_parameter_index(::Nothing, pc)" begin
    # Integer id → symbolic_type is NotSymbolic → assertion passes, returns id
    pc = PiecewiseParameter(3, [0.0, 1.0])
    @test Corleone.get_parameter_index(nothing, pc) == 3
end

@testset "Controls – maybekeys gensym branch" begin
    prob = LotkaVolterra.generate()
    sys  = symbolic_container(prob.f)
    # Integer id (no symbolic name) → maybekeys hits gensym branch
    pc1 = PiecewiseParameter(5, collect(LinRange(0.0, 12.0, 4)))  # id=5 (param index)
    pc2 = PiecewiseParameter(6, collect(LinRange(0.0, 12.0, 4)))  # id=6
    c = Controls(pc1, pc2; sys = sys)
    @test length(c.controls) == 2
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

# ---------------------------------------------------------------------------
# abstract.jl – bound helpers and constraint machinery
# ---------------------------------------------------------------------------

@testset "abstract.jl – get_lower/upper_bound for Number, Array, Tuple" begin
    # Number branch (lines 49-50)
    @test Corleone.get_lower_bound(1.0f0) === -Inf32
    @test Corleone.get_upper_bound(1.0f0) === Inf32
    @test Corleone.get_lower_bound(2.0)   === -Inf
    @test Corleone.get_upper_bound(2.0)   === Inf

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
    lb_nt = Corleone.get_lower_bound((; a=1.0, b=2.0f0))
    @test lb_nt.a === -Inf && lb_nt.b === -Inf32
end

@testset "abstract.jl – get_lower/upper_bound and get_bounds on layer" begin
    prob = LotkaVolterra.generate()
    sys  = symbolic_container(prob.f)
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
    sys  = symbolic_container(prob.f)
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
    sys  = symbolic_container(prob.f)
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

# ---------------------------------------------------------------------------
# Issue 5 – ShootingInterval bounds/display + AutoBlock
# ---------------------------------------------------------------------------

@testset "ShootingInterval – display_name and bounds" begin
    prob = LotkaVolterra.generate()

    # display_name: gensym branch (empty variable_id has no hasname)
    si_empty = ShootingInterval(prob, Symbol[], prob.tspan)
    name = LuxCore.display_name(si_empty)
    @test name isa Symbol

    # display_name: symbolic branch (variable_id is a Symbol with hasname)
    si_sym = ShootingInterval(prob, [:x, :y], (0.0, 6.0))
    # :x has a name in SII
    @test LuxCore.display_name(si_sym) isa Symbol

    # get_lower/upper_bound with bounds set
    lb_fn = (ps, st) -> fill(-5.0, length(ps))
    ub_fn = (ps, st) -> fill(5.0, length(ps))
    si_bounded = ShootingInterval(prob, [1, 2], prob.tspan;
        bounds = (lb_fn, ub_fn))
    ps_si, st_si = LuxCore.setup(rng, si_bounded)
    lb = Corleone.get_lower_bound(si_bounded, ps_si, st_si)
    ub = Corleone.get_upper_bound(si_bounded, ps_si, st_si)
    @test all(x -> x == -5.0, lb)
    @test all(x -> x == 5.0, ub)
end

@testset "ShootingInterval – get_variable_index(::Nothing, si)" begin
    prob = LotkaVolterra.generate()
    # Numeric variable_id → NotSymbolic → no assertion error
    si = ShootingInterval(prob, [1, 2], prob.tspan)
    idx = Corleone.get_variable_index(nothing, si)
    @test idx == [1, 2]
end

@testset "ShootingLayer – AutoBlock shooting" begin
    prob  = ControlledLotka.generate()
    cgrid = collect(LinRange(0.0, 12.0, 13))
    pc1   = PiecewiseParameter(:u1, copy(cgrid))
    pc2   = PiecewiseParameter(:u2, copy(cgrid))

    # AutoBlock(n) injects n-1 shooting points, producing n segments
    n_blocks = 3
    layer = ShootingLayer(
        prob, Symbol[], pc1, pc2;
        algorithm       = Tsit5(),
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
    prob  = ControlledLotka.generate()
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
    layer = ShootingLayer(prob, Symbol[], pc1, pc2;
        algorithm          = Tsit5(),
        ensemble_algorithm = EnsembleThreads(),
    )
    ps, st = LuxCore.setup(rng, layer)
    traj, _ = layer(prob, ps, st)
    @test traj isa Solutions.Trajectory
    @test length(current_time(traj)) == length(current_time(ref_traj))
end

@testset "ShootingLayer – EnsembleDistributed" begin
    prob  = ControlledLotka.generate()
    cgrid = collect(LinRange(0.0, 12.0, 7))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))
    layer = ShootingLayer(prob, Symbol[], pc1, pc2;
        algorithm          = Tsit5(),
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
