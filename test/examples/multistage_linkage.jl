using Corleone
using ComponentArrays
using ForwardDiff
using Ipopt
using LinearAlgebra
using Lux
using LuxCore
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationMOI
using Random
using SciMLBase
using SciMLSensitivity
using SciMLSensitivity.Zygote
using StableRNGs
using Test

# `StageLinkage` links consecutive optimal-control stages by an *interface state* that is
# itself a free optimization variable, and enforces a matching residual
#     Δx_min ≤ u0 - m(x_f_prev, θ) ≤ Δx_max
# instead of substituting the next phase's initial state. These tests cover the identity,
# affine and general-map constructors, the bounds/parameter bookkeeping, the collected
# linkage residuals, and the two multiple-shooting interaction bugs the linkage exposes
# (a non-final multiple-shooting stage must honour the incoming interface state and its
# matching residuals must be collected).

rng = StableRNG(7)

function _linkage_rhs!(du, u, p, t)
    du[1] = p[1] * u[1]
    return
end

prob1 = ODEProblem(_linkage_rhs!, [1.0], (0.0, 1.0), [0.05])
prob2 = ODEProblem(_linkage_rhs!, [1.0], (0.0, 1.0), [0.07])

@testset "Identity linkage" begin
    stage1 = SingleShootingLayer(prob1, Tsit5())
    stage2 = SingleShootingLayer(prob2, Tsit5())
    linkage = StageLinkage([1.0]; bounds_u0 = ([0.5], [2.0]))
    chain = Chain(stage1, linkage, stage2)

    ps, st = LuxCore.setup(rng, chain)

    # The linkage owns a tunable interface state; the θ block is empty for identities.
    @test ps.layer_2.u0 == [1.0]
    @test isempty(ps.layer_2.θ)
    @test Corleone.get_number_of_linkage_constraints(chain) == 1
    lb, ub = Corleone.get_bounds(chain)
    @test lb.layer_2.u0 == [0.5]
    @test ub.layer_2.u0 == [2.0]

    # Default residual bounds are an exact linkage.
    dynprob = Corleone.CorleoneDynamicOptProblem(chain, :x₁)
    @test dynprob.lcons == [0.0]
    @test dynprob.ucons == [0.0]

    # The residual is u0 - x_f(stage1); writing it must not leave NaNs behind.
    res = fill(NaN, 1)
    dynprob.constraints(res, ps, st)
    @test all(isfinite, res)
    xf = last(stage1(nothing, ps.layer_1, st.layer_1)[1].u)
    @test res[1] ≈ 1.0 - xf[1]

    # The next stage starts at the linkage's interface state, independent of stage 1.
    ps_shift = deepcopy(ps)
    ps_shift.layer_2.u0 .= 1.5
    traj, _ = chain(nothing, ps_shift, st)
    sim2, _ = stage2([1.5], ps_shift.layer_3, st.layer_3)
    @test last(traj.u) ≈ last(sim2.u)
end

@testset "Affine linkage" begin
    stage1 = SingleShootingLayer(prob1, Tsit5())
    stage2 = SingleShootingLayer(prob2, Tsit5())
    # x_next = A x_f + B θ + c with a one-dimensional state and one jump parameter.
    A = reshape([1.0], 1, 1)
    B = reshape([1.0], 1, 1)
    c = [0.0]
    linkage = StageLinkage(
        A, B, c; u0 = [1.0], θ = [0.0],
        bounds_u0 = ([0.5], [2.0]), bounds_θ = ([-1.0], [1.0]),
        bounds = ([-0.1], [0.1]),
    )
    chain = Chain(stage1, linkage, stage2)
    ps, st = LuxCore.setup(rng, chain)

    @test ps.layer_2.u0 == [1.0]
    @test ps.layer_2.θ == [0.0]
    @test Corleone.get_number_of_linkage_constraints(chain) == 1
    lb, ub = Corleone.get_bounds(chain)
    @test lb.layer_2.θ == [-1.0]
    @test ub.layer_2.θ == [1.0]

    dynprob = Corleone.CorleoneDynamicOptProblem(chain, :x₁)
    @test dynprob.lcons == [-0.1]
    @test dynprob.ucons == [0.1]

    res = fill(NaN, 1)
    dynprob.constraints(res, ps, st)
    xf = last(stage1(nothing, ps.layer_1, st.layer_1)[1].u)
    @test res[1] ≈ 1.0 - (xf[1] + 0.0)
end

@testset "General-map linkage (reactor-style mixing)" begin
    stage1 = SingleShootingLayer(prob1, Tsit5())
    stage2 = SingleShootingLayer(prob2, Tsit5())
    # Dilution-like map that is affine in the state but *nonlinear* in θ.
    dil = (x, θ) -> [x[1] / (1.0 + θ[1])]
    linkage = StageLinkage(
        dil; u0 = [1.0], θ = [0.0], bounds_θ = ([0.0], [1.0]), bounds_u0 = ([0.0], [2.0])
    )
    chain = Chain(stage1, linkage, stage2)
    ps, st = LuxCore.setup(rng, chain)

    xf = last(stage1(nothing, ps.layer_1, st.layer_1)[1].u)
    @test Corleone.linkage_residual(linkage, ps.layer_2, stage1(nothing, ps.layer_1, st.layer_1)[1])[1] ≈
        1.0 - xf[1]

    ps_shift = deepcopy(ps)
    ps_shift.layer_2.θ .= 1.0
    r = Corleone.linkage_residual(linkage, ps_shift.layer_2, stage1(nothing, ps.layer_1, st.layer_1)[1])
    @test r[1] ≈ 1.0 - xf[1] / 2.0
end

@testset "MultipleShooting stage honours incoming interface state" begin
    # A non-first multiple-shooting phase must start its first interval at the incoming
    # interface state (previously `_parallel_solve` dropped it). Each shooting interval is
    # solved independently, so only the first segment of the merged trajectory changes.
    ms = MultipleShootingLayer(prob2, Tsit5(), 0.5; tunable_ic = [1])
    straight, _ = ms([2.0], LuxCore.initialparameters(rng, ms), LuxCore.initialstates(rng, ms))
    via_nothing, _ = ms(nothing, LuxCore.initialparameters(rng, ms), LuxCore.initialstates(rng, ms))
    @test first(straight.u)[1] == 2.0
    @test first(via_nothing.u)[1] == 1.0
    @test !isapprox(first(straight.u)[1], first(via_nothing.u)[1])
    # The first shooting residual now reflects the incoming state rather than the default.
    n = Corleone.get_number_of_shooting_constraints(ms)
    res = fill(NaN, n)
    Corleone.shooting_constraints!(view(res, 1:n), straight)
    @test all(isfinite, res)
end

@testset "Linkage residuals collected for MultipleShooting chains" begin
    stage1 = SingleShootingLayer(prob1, Tsit5())
    ms2 = MultipleShootingLayer(prob2, Tsit5(), 0.5; tunable_ic = [1])
    linkage = StageLinkage([1.0]; bounds_u0 = ([0.5], [2.0]))
    chain = Chain(stage1, linkage, ms2)
    ps, st = LuxCore.setup(rng, chain)

    n_shoot = Corleone.get_number_of_shooting_constraints(chain)
    n_link = Corleone.get_number_of_linkage_constraints(chain)
    @test n_link == 1

    dynprob = Corleone.CorleoneDynamicOptProblem(chain, :x₁)
    @test length(dynprob.lcons) == n_shoot + n_link
    res = fill(NaN, length(dynprob.lcons))
    dynprob.constraints(res, ps, st)
    # Every row, including the multiple-shooting matching rows of a *non-final* stage,
    # must be written.
    @test all(isfinite, res)

    # Moving the interface state moves the linkage residual and the matching residual of
    # the multiple-shooting stage together (both start from the interface state).
    ps_shift = deepcopy(ps)
    ps_shift.layer_2.u0 .= 1.5
    res_shift = fill(NaN, length(dynprob.lcons))
    dynprob.constraints(res_shift, ps_shift, st)
    @test res_shift[end] ≈ 1.5 - last(stage1(nothing, ps.layer_1, st.layer_1)[1].u)[1]
end

@testset "Per-stage objective accumulation" begin
    stage1 = SingleShootingLayer(prob1, Tsit5())
    stage2 = SingleShootingLayer(prob2, Tsit5())
    linkage = StageLinkage([1.0]; bounds_u0 = ([0.5], [2.0]))
    chain = Chain(stage1, linkage, stage2)
    ps, st = LuxCore.setup(rng, chain)

    # A bare symbol still targets the final stage.
    dyn_last = Corleone.CorleoneDynamicOptProblem(chain, :x₁)
    @test dyn_last.objective(ps, st) ≈ last(chain(nothing, ps, st)[1].u)[1]

    # A tuple sums per-stage terminal values; this removes the manual
    # "carry the accumulated cost in a state" plumbing.
    dyn_sum = Corleone.CorleoneDynamicOptProblem(chain, ((:x₁, 1), (:x₁, 2)))
    trajs = Corleone._forward_collect_stage_trajs(chain, ps, st)
    expected = last(trajs[1].u)[1] + last(trajs[2].u)[1]
    @test dyn_sum.objective(ps, st) ≈ expected
end

@testset "Multi-linkage chain constraint assembly" begin
    # A chain mixing single- and multiple-shooting phases with two linkages must order
    # and write every constraint row (path, shooting and linkage) of every phase.
    stage1 = SingleShootingLayer(prob1, Tsit5())
    ms2 = MultipleShootingLayer(prob2, Tsit5(), 0.5; tunable_ic = [1])
    stage3 = SingleShootingLayer(prob2, Tsit5())
    link12 = StageLinkage([1.0]; bounds_u0 = ([0.5], [2.0]))
    link23 = StageLinkage([1.0]; bounds_u0 = ([0.5], [2.0]))
    chain = Chain(stage1, link12, ms2, link23, stage3)
    ps, st = LuxCore.setup(rng, chain)

    # Path constraints are ordered by stage; each stage only couples to the previous one.
    dynprob = Corleone.CorleoneDynamicOptProblem(
        chain, :x₁,
        (:x₁ => (t = [1.0], bounds = (0.0, Inf))),
        ((:x₁, 3) => (t = [1.0], bounds = (0.0, Inf))),
        ((:x₁, 3) => (t = [0.0], bounds = (0.0, Inf))),
    )
    @test Corleone.get_number_of_linkage_constraints(chain) == 2
    res = fill(NaN, length(dynprob.lcons))
    dynprob.constraints(res, ps, st)
    @test all(isfinite, res)
end

# --- Linkage sparsity / block-diagonal Hessian ------------------------------------
# Linking phases by an interface-state constraint (rather than substituting the next
# phase's initial state) keeps the stage coupling *linear*: each linkage row only touches
# one phase plus the interface, and the interface state never enters the earlier phases.
# The constraint Jacobian is therefore block-banded and the Hessian of the Lagrangian is
# block-diagonal -- exactly the structure a sparse/block SQP solver wants. Writing the
# same chain with `StageTransition`s produces a dense, fully coupled Hessian instead.

function _struct_rhs_a!(du, u, p, t)
    du[1] = p[1] * u[1]
    du[2] = p[2] * u[2]
    return
end
function _struct_rhs_b!(du, u, p, t)
    du[1] = -p[1] * u[1]
    du[2] = p[2] * u[1]
    return
end

const _struct_u0 = [1.0, 2.0]
const _struct_p = [1.0, 1.0]
const _struct_cons = (
    (:x₁, 2) => (t = [1.0], bounds = (0.0, 2.0)),
    (:x₂, 2) => (t = [1.0], bounds = (0.0, 2.0)),
)

# Return the constraint Jacobian and the Lagrangian Hessian of the two-stage chain with
# the given middle layer, evaluated at the initial point.
function _struct_derivs(middle)
    stage1 = SingleShootingLayer(
        ODEProblem(_struct_rhs_a!, _struct_u0, (0.0, 1.0), _struct_p), Tsit5();
        bounds_p = ([-10.0, -10.0], [10.0, 10.0]),
    )
    stage2 = SingleShootingLayer(
        ODEProblem(_struct_rhs_b!, _struct_u0, (0.0, 1.0), _struct_p), Tsit5();
        bounds_p = ([-10.0, -10.0], [10.0, 10.0]),
    )
    chain = Chain(stage1, middle, stage2)
    optprob = OptimizationProblem(
        chain, AutoForwardDiff(), Val(:ComponentArrays);
        loss = ((:x₁, 2), (:x₂, 2)), constraints = _struct_cons,
    )
    x0 = optprob.u0
    m = length(optprob.lcons)
    consf = x -> begin
        res = zeros(eltype(x), m)
        optprob.f.cons(res, x, optprob.p)
        return res
    end
    J = ForwardDiff.jacobian(consf, x0)
    # Lagrangian with unit multipliers: enough to expose all Hessian coupling.
    L = x -> optprob.f(x, optprob.p) + sum(consf(x))
    return J, ForwardDiff.hessian(L, x0)
end

# Largest absolute cross-block entry, splitting the variables after the first stage's
# two parameters (the interface state and the second stage follow).
function _struct_coupling(H)
    n1 = 2
    return maximum(abs, H[1:n1, (n1 + 1):end]) + maximum(abs, H[(n1 + 1):end, 1:n1])
end

@testset "Linkage yields block-diagonal Lagrangian Hessian" begin
    jump = StageTransition((traj, p) -> last(traj.u))
    link = StageLinkage([2.0, 3.0]; bounds_u0 = ([-10.0, -10.0], [10.0, 10.0]))

    J_jump, H_jump = _struct_derivs(jump)
    J_link, H_link = _struct_derivs(link)

    # Substitution couples phase 1 and phase 2 densely; the linkage decouples them.
    @test _struct_coupling(H_jump) > 1.0
    @test _struct_coupling(H_link) ≈ 0.0 atol = 1.0e-10

    # The linkage Jacobian is also sparser than the substitution Jacobian.
    @test count(!iszero, J_link) / length(J_link) <
        count(!iszero, J_jump) / length(J_jump)
end

# --- AD engine compatibility ------------------------------------------------------
# ForwardDiff and Zygote are the supported differentiation backends for the multistage
# pipeline (ReverseDiff is currently broken). A linkage chain must build an
# `OptimizationProblem` under both and reach a finite solution.
function make_linkage_op(AD)
    sensealg = AD == AutoZygote() ? ForwardDiffSensitivity() : SciMLBase.NoAD()
    p1 = ODEProblem(_linkage_rhs!, [1.0], (0.0, 1.0), [0.05]; sensealg)
    p2 = ODEProblem(_linkage_rhs!, [1.0], (0.0, 1.0), [0.07]; sensealg)
    stage1 = SingleShootingLayer(p1, Tsit5(); bounds_p = ([0.01], [0.2]))
    stage2 = SingleShootingLayer(p2, Tsit5(); bounds_p = ([0.01], [0.2]))
    linkage = StageLinkage([1.0]; bounds_u0 = ([0.5], [2.0]))
    chain = Chain(stage1, linkage, stage2)
    return OptimizationProblem(
        chain, AD, Val(:ComponentArrays);
        loss = ((:x₁, 1), (:x₁, 2)),
        constraints = (:x₁ => (t = [1.0], bounds = (0.0, 3.0)),),
    )
end

@testset "AD compatibility: $AD" for AD in (AutoForwardDiff(), AutoZygote())
    optprob_ad = make_linkage_op(AD)
    @test isfinite(optprob_ad.f(optprob_ad.u0, optprob_ad.p))
    sol_ad = solve(
        optprob_ad, Ipopt.Optimizer(); max_iter = 50, print_level = 0,
        hessian_approximation = "limited-memory",
    )
    @test SciMLBase.successful_retcode(sol_ad)
    @test isfinite(sol_ad.objective)
end
