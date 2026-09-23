# ---------------------------------------------------------------------------
# DynamicProblemLayer – wraps a ShootingLayer together with an objective and
# (optional) constraints, both built as parser-based DynamicFunctionLayers
# from symbolic expressions over the shooting layer's system.
#
# API:
#   DynamicProblemLayer(layer::ShootingLayer, obj, constraints...; kwargs...)
#   Corleone.evaluate_objective(prob, x::AbstractDEProblem, ps, st)   -> (obj, st)
#   Corleone.evaluate_constraints(prob, x, ps, st)                   -> (cons, st)
#     (x::AbstractDEProblem for out-of-place, (res, x)::Tuple for in-place;
#      cons === nothing, st unchanged when prob.constraints === nothing)
#   Corleone.predict(prob, x::AbstractDEProblem, ps, st)             -> (traj, st)
#   (prob::DynamicProblemLayer)(x, ps, st)                           -> ((obj, cons), st)
# ---------------------------------------------------------------------------

function _shooting_layer_prob(; n = 6)
    prob = ControlledLotka.generate()
    cgrid = collect(LinRange(0.0, 12.0, n))
    pc1 = PiecewiseParameter(:u1, copy(cgrid))
    pc2 = PiecewiseParameter(:u2, copy(cgrid))
    layer = ShootingLayer(prob, Symbol[], pc1, pc2; algorithm = Tsit5())
    return prob, layer
end

const OBJ_EXPR = :(x(3.0) + y(6.0))
const CONS_EXPRS = (:(x(3.0) - 1.0 == 0), :(x(5.0) * y(5.0) >= 3.0))

@testset "construction and LuxCore.setup" begin
    de_prob, shooting = _shooting_layer_prob()

    @testset "no constraints given: prob.constraints === nothing" begin
        prob = DynamicProblemLayer(shooting, OBJ_EXPR; rng = rng)
        @test prob.constraints === nothing
        @test prob.objective isa DynamicFunctionLayer

        ps, st = LuxCore.setup(rng, prob)
        @test ps isa NamedTuple{(:shooting, :objective, :constraints)}
        @test st isa NamedTuple{(:shooting, :objective, :constraints)}
        @test ps.constraints == (;)
        @test st.constraints == (;)
    end

    @testset "constraints given: prob.constraints is a DynamicFunctionLayer" begin
        prob = DynamicProblemLayer(shooting, OBJ_EXPR, CONS_EXPRS...; rng = rng)
        @test prob.constraints isa DynamicFunctionLayer

        ps, st = LuxCore.setup(rng, prob)
        @test haskey(st.constraints, :saveat)
        @test st.constraints.lb == [0.0, -Inf]
        @test st.constraints.ub == [0.0, 0.0]
    end
end

@testset "evaluate_objective matches a manual remake + shooting + objective run" begin
    de_prob, shooting = _shooting_layer_prob()
    prob = DynamicProblemLayer(shooting, OBJ_EXPR, CONS_EXPRS...; rng = rng)
    ps, st = LuxCore.setup(rng, prob)

    obj, st′ = Corleone.evaluate_objective(prob, de_prob, ps, st)

    de = remake(de_prob, saveat = st.objective.saveat)
    expected_traj, expected_shooting_st = shooting(de, ps.shooting, st.shooting)
    expected_obj, expected_objective_st = prob.objective(expected_traj, ps.objective, st.objective)

    @test obj == expected_obj
    @test st′.shooting == expected_shooting_st
    @test st′.objective == expected_objective_st
    @test st′.constraints == st.constraints  # untouched
end

@testset "evaluate_constraints" begin
    de_prob, shooting = _shooting_layer_prob()
    prob = DynamicProblemLayer(shooting, OBJ_EXPR, CONS_EXPRS...; rng = rng)
    ps, st = LuxCore.setup(rng, prob)

    @testset "out-of-place (x::AbstractDEProblem) matches a manual run" begin
        cons, st′ = Corleone.evaluate_constraints(prob, de_prob, ps, st)

        de = remake(de_prob, saveat = st.constraints.saveat)
        expected_traj, expected_shooting_st = shooting(de, ps.shooting, st.shooting)
        expected_cons, expected_constraints_st = prob.constraints(expected_traj, ps.constraints, st.constraints)

        @test cons == expected_cons
        @test st′.shooting == expected_shooting_st
        @test st′.constraints == expected_constraints_st
        @test st′.objective == st.objective  # untouched
    end

    @testset "in-place ((res, x)::Tuple) writes into res and matches a manual run" begin
        res = zeros(2)
        cons, st′ = Corleone.evaluate_constraints(prob, (res, de_prob), ps, st)

        de = remake(de_prob, saveat = st.constraints.saveat)
        expected_traj, _ = shooting(de, ps.shooting, st.shooting)
        expected_res = zeros(2)
        expected_cons, expected_constraints_st = prob.constraints(expected_res, expected_traj, ps.constraints, st.constraints)

        @test cons === res
        @test res == expected_res
        @test st′.constraints == expected_constraints_st
    end

    @testset "prob.constraints === nothing short-circuits without touching x or st" begin
        prob_no_cons = DynamicProblemLayer(shooting, OBJ_EXPR; rng = rng)
        ps_nc, st_nc = LuxCore.setup(rng, prob_no_cons)

        cons, st′ = Corleone.evaluate_constraints(prob_no_cons, de_prob, ps_nc, st_nc)
        @test cons === nothing
        @test st′ == st_nc

        res = fill(NaN, 1)
        cons_ip, st′_ip = Corleone.evaluate_constraints(prob_no_cons, (res, de_prob), ps_nc, st_nc)
        @test cons_ip === nothing
        @test st′_ip == st_nc
        @test all(isnan, res)  # never written to
    end
end

@testset "predict remakes with the union of objective/constraints saveats and matching tspan" begin
    de_prob, shooting = _shooting_layer_prob()
    prob = DynamicProblemLayer(shooting, OBJ_EXPR, CONS_EXPRS...; rng = rng)
    ps, st = LuxCore.setup(rng, prob)

    traj, st′ = Corleone.predict(prob, de_prob, ps, st)
    @test traj isa Solutions.Trajectory

    combined_saveat = (unique! ∘ sort!)(vcat(st.objective.saveat, st.constraints.saveat))
    de = remake(de_prob, saveat = combined_saveat, tspan = extrema(combined_saveat))
    expected_traj, expected_shooting_st = shooting(de, ps.shooting, st.shooting)

    @test extrema(current_time(traj)) == extrema(current_time(expected_traj))
    @test st′.shooting == expected_shooting_st
    @test st′.objective == st.objective
    @test st′.constraints == st.constraints
end

@testset "callable (prob::DynamicProblemLayer)(x, ps, st) combines evaluate_objective/evaluate_constraints" begin
    de_prob, shooting = _shooting_layer_prob()
    prob = DynamicProblemLayer(shooting, OBJ_EXPR, CONS_EXPRS...; rng = rng)
    ps, st = LuxCore.setup(rng, prob)

    (obj, cons), st′ = prob(de_prob, ps, st)

    expected_obj, st_after_obj = Corleone.evaluate_objective(prob, de_prob, ps, st)
    expected_cons, expected_st = Corleone.evaluate_constraints(prob, de_prob, ps, st_after_obj)

    @test obj == expected_obj
    @test cons == expected_cons
    @test st′ == expected_st
end
