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
    si_bounded = ShootingInterval(
        prob, [1, 2], prob.tspan;
        bounds = (lb_fn, ub_fn)
    )
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
