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
    dict_cache = Dict{Any, Int}()
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
