using Corleone
using ComponentArrays
using Ipopt
using Lux
using LuxCore
using Optimization
using OptimizationMOI
using OrdinaryDiffEqTsit5
using Random
using SciMLBase
using SciMLSensitivity
using SciMLSensitivity.Zygote
using Test

# Two-stage batch reactor optimal control, following Hirmajer & Fikar, "Optimal
# Control of a Two-Stage Reactor System", Chem. Pap. 60(5), 2006,
# DOI: 10.2478/s11696-006-0069-x.
#
# The first (non-isothermal) stage runs A -> B -> C with the reactor temperature as the
# control. At the switching time a volume `S` of a B solution (concentration `c_b^s`) is
# added, diluting the mixture, and the second (isothermal) stage runs three parallel
# reactions B -> D, B -> E and 2B -> F. We maximise the total amount of D at the final
# time `t_P`, i.e. `J0 = V2 * c_D(t_P)`, subject to a minimum final concentration of D
# and a maximum processing time.
#
# Both stages are formulated in normalized time `tau in [0, 1]` and their right-hand
# sides are scaled by the (optimised) stage durations `T1` and `T2`. The temperature
# control is piecewise constant on a uniform grid of the first stage. The volume of added
# B solution enters as the parameter of the `StageLinkage`, which applies the mixing
# (dilution) relations of the paper and couples the two phases through an interface state.

# --- Stage 1: non-isothermal, A -> B -> C, temperature `T` as control, duration `T1` ---
function stage1!(du, u, p, t)
    A, B, C, D, E, F, D_amount, V2 = u
    T1, T = p
    k1 = 0.0444 * exp(-2500.0 / T)
    k2 = 6889.0 * exp(-5000.0 / T)
    du[1] = T1 * (-2 * k1 * A^2)
    du[2] = T1 * (k1 * A^2 - k2 * B)
    du[3] = T1 * (k2 * B)
    du[4] = 0.0
    du[5] = 0.0
    du[6] = 0.0
    du[7] = 0.0 # amount of D stays constant (c_D = 0 throughout stage 1)
    du[8] = 0.0 # stage-2 volume is set at the switch
    return
end

# --- Stage 2: isothermal, B -> D, B -> E, 2B -> F, duration `T2` ---
function stage2!(du, u, p, t)
    A, B, C, D, E, F, D_amount, V2 = u
    T2 = p[1]
    du[1] = 0.0
    du[2] = T2 * (-0.02 * B - 0.05 * B - 0.00008 * B^2)
    du[3] = 0.0
    du[4] = T2 * (0.02 * B)
    du[5] = T2 * (0.05 * B)
    du[6] = T2 * (0.00004 * B^2)
    du[7] = T2 * (V2 * 0.02 * B) # d(D_amount)/dt = V2 * d(c_D)/dt
    du[8] = 0.0
    return
end

# --- Linkage: add volume `S` of a B solution, dilute the mixture to V2 = V1 + S ---
# Mixing equations of the paper:
#   V2 c_A(t+) = V1 c_A(t-),  V2 c_B(t+) = V1 c_B(t-) + S c_B^s,  V2 c_C(t+) = V1 c_C(t-)
# The 7th state carries the amount of D (V * c_D), which is unchanged by the mixing; the
# 8th state carries the stage-2 volume.
const V1 = 0.1
const c_b_s = 600.0
function switch_at_ts(x, θ)
    A, B, C, D, E, F, D_amount, V2 = x
    S = θ[1]
    V2_new = V1 + S
    return [
        V1 * A / V2_new, (V1 * B + S * c_b_s) / V2_new, V1 * C / V2_new,
        V1 * D / V2_new, V1 * E / V2_new, V1 * F / V2_new,
        V1 * D,    # amount of D is preserved across the switch
        V2_new,    # carry the stage-2 volume
    ]
end

u0 = [2000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

const t_p = 180.0  # maximum/desired processing time [min]
const c_d_w = 150.0 # minimum final concentration of D [mol/m^3]

# Initial durations: 5 first-stage control intervals of 15 min plus a 15 min second stage.
T1_init = 5.0 * 15.0
T2_init = 15.0

prob1 = ODEProblem(stage1!, u0, (0.0, 1.0), [T1_init, 350.0], sensealg = SciMLBase.NoAD())
prob2 = ODEProblem(stage2!, u0, (0.0, 1.0), [T2_init], sensealg = SciMLBase.NoAD())

# Piecewise-constant temperature on 5 uniform intervals of the first stage (normalized
# time). Bounds of the control are the reactor temperature limits.
grid = collect(0.0:0.2:0.8)

stage1 = ShootingLayer(
    prob1, Tsit5(),
    Tunable(1; kind = Val(:parameter), lowerbound = 10.0, upperbound = 170.0) => nothing,
    Tunable(2; kind = Val(:parameter), lowerbound = 298.0, upperbound = 398.0) => grid[2:end],
)
stage2 = ShootingLayer(
    prob2, Tsit5(),
    Tunable(1; kind = Val(:parameter), lowerbound = 10.0, upperbound = 170.0) => nothing,
)

# The linkage owns the second stage's initial state, initialized consistently with the
# first stage's forward solution for `S = 0.1`; the added volume is bounded to [0, 0.1].
let ps1, st1
    ps1, st1 = LuxCore.setup(Random.default_rng(), stage1)
    traj1, _ = stage1(nothing, ps1, st1)
    global u0_linkage = switch_at_ts(last(traj1.u), [0.1])
end

linkage = StageLinkage(
    switch_at_ts; u0 = u0_linkage, θ = [0.1], bounds_θ = ([0.0], [0.1])
)
chain = Chain(stage1, linkage, stage2)

# Structure of the composed problem: 1 duration + 5 temperature controls for stage 1,
# 1 interface state + 1 volume parameter for the linkage, 1 duration for stage 2.
ps, st = LuxCore.setup(Random.default_rng(), chain)

# --- Type stability ---------------------------------------------------------------

@testset "multistage chain API" begin
    # The new recursive bounds/collection API is not fully inferred, so these assertions
    # check the returned *kind* of container rather than exact return types.
    @test LuxCore.parameterlength(stage1) isa Int
    @test LuxCore.parameterlength(stage2) isa Int
    @test LuxCore.parameterlength(linkage) isa Int
    @test Corleone.get_bounds(stage1, ps.layer_1, st.layer_1) isa Tuple
    @test Corleone.get_bounds(stage2, ps.layer_3, st.layer_3) isa Tuple
    @test Corleone.get_bounds(chain, ps, st) isa Tuple
    @test Corleone._multistage_stages(chain) isa Vector
    @test Corleone._multistage_forward(chain, ps, st) isa Tuple
    @test Corleone.get_number_of_shooting_constraints(chain, ps, st) isa Int
    @test chain(nothing, ps, st) isa Tuple
end

# Parameter bookkeeping: stage 1 (1 duration + 5 temperature controls), the linkage
# (8 interface-state components + 1 added volume) and stage 2 (1 duration).
@test LuxCore.parameterlength(chain) ==
    LuxCore.parameterlength(stage1) + LuxCore.parameterlength(linkage) + LuxCore.parameterlength(stage2)
lb, ub = Corleone.get_bounds(chain, ps, st)
p = ComponentArray(ps)
@test length(ComponentArray(lb)) == length(p)
@test length(ComponentArray(ub)) == length(p)
@test ComponentArray(lb).layer_2.θ == [0.0]
@test ComponentArray(ub).layer_2.θ == [0.1]
@test Corleone.get_number_of_shooting_constraints(chain, ps, st) == 0
@test Corleone.get_number_of_linkage_constraints(chain) == 8

# The linkage applies the mixing relations: dilution by V1/V2 and addition of B.
A, B, C, D, E, F, D_amount, V2 = last((stage1(nothing, ps.layer_1, st.layer_1))[1].u)
expec = let S = ps.layer_2.θ[1], V2n = V1 + S
    [
        V1 * A / V2n, (V1 * B + S * c_b_s) / V2n, V1 * C / V2n, V1 * D / V2n,
        V1 * E / V2n, V1 * F / V2n, V1 * D, V2n,
    ]
end
@test Corleone.linkage_residual(linkage, ps.layer_2, (stage1(nothing, ps.layer_1, st.layer_1))[1]) ≈
    ps.layer_2.u0 - expec

# Maximum amount of D is wanted, so the objective is minimised for -D_amount (state 7).
# Constraint J2: the two stage durations must fit within the processing time, i.e.
#   T1 + T2 <= t_P  (a parameter constraint coupling the two stages).
duration_constraint = (ps -> [
    ps.layer_1.parameter.parameters.p₁.parameter.ps[1] +
        ps.layer_3.parameter.parameters.p₁.parameter.ps[1] - t_p
]) => (lb = [-Inf], ub = [0.0])

# Constraint J1: c_D(t_P) >= c_D^w. A terminal constraint is just a path constraint
# evaluated at the final value only, so it is imposed on the final stage at t = 1.
optprob = OptimizationProblem(
    chain, AutoForwardDiff(), Val(:ComponentArrays);
    loss = :(-u₇), # maximise J0 = V2 * c_D(t_P) = D_amount(t_P)
    constraints = ((:u₄, 2) => (t = [1.0], bounds = (c_d_w, Inf)),),
    parameter_constraints = duration_constraint,
)
@test length(optprob.u0) == LuxCore.parameterlength(chain)
# One terminal path constraint, eight exact linkage residuals, one parameter constraint.
@test optprob.lcons == [c_d_w; zeros(8); -Inf]
@test optprob.ucons == [Inf; zeros(8); 0.0]

sol = solve(
    optprob, Ipopt.Optimizer(); max_iter = 500, tol = 1.0e-8,
    hessian_approximation = "limited-memory", print_level = 0,
)

@test SciMLBase.successful_retcode(sol)
# J0 = V2 * c_D(t_P) = 25.51 mol, matching the paper's 25.5365 for P = 6.
@test isapprox(-sol.objective, 25.5148, atol = 1.0e-3)

p_opt = sol.u .+ zero(p)

# Constraint J2 is active: T1 + T2 = t_P.
@test p_opt.layer_1.parameter.parameters.p₁.parameter.ps[1] +
    p_opt.layer_3.parameter.parameters.p₁.parameter.ps[1] ≈ t_p atol = 1.0e-6
# Switching time ts = T1 ≈ 106 min (paper: 106.04) and added volume S ≈ 0.070 m^3
# (paper: 0.0702).
@test isapprox(p_opt.layer_1.parameter.parameters.p₁.parameter.ps[1], 105.98, atol = 1.0e-2)
@test isapprox(p_opt.layer_2.θ[1], 0.0701, atol = 1.0e-3)

# Constraint J1 is active: c_D(t_P) = 150 mol/m^3 and J0 = V2 * c_D(t_P).
traj, _ = chain(nothing, p_opt, st)
u_end = last(traj.u)
@test isapprox(u_end[4], c_d_w, atol = 1.0e-3)
@test isapprox(u_end[7], u_end[4] * u_end[8], atol = 1.0e-6)
@test isapprox(u_end[7], -sol.objective, atol = 1.0e-6)
@test -sol.objective ≈ c_d_w * (V1 + p_opt.layer_2.θ[1]) atol = 1.0e-3

# --- AD engine compatibility ------------------------------------------------------
# ForwardDiff and Zygote are the currently supported differentiation backends for the
# multistage / single-shooting pipeline; ReverseDiff is currently broken.
#
# `ForwardDiffSensitivity` is required for the Zygote objective, matching the rule in
# `lotka_oc.jl`. Each backend must build the problem, evaluate the objective and propagate a
# gradient through the shooting layers.
function make_reactor_op(AD)
    sensealg = AD == AutoZygote() ? ForwardDiffSensitivity() : SciMLBase.NoAD()
    prob1_ad = ODEProblem(stage1!, u0, (0.0, 1.0), [T1_init, 350.0]; sensealg)
    prob2_ad = ODEProblem(stage2!, u0, (0.0, 1.0), [T2_init]; sensealg)
    stage1_ad = ShootingLayer(
        prob1_ad, Tsit5(),
        Tunable(1; kind = Val(:parameter), lowerbound = 10.0, upperbound = 170.0) => nothing,
        Tunable(2; kind = Val(:parameter), lowerbound = 298.0, upperbound = 398.0) => grid[2:end],
    )
    stage2_ad = ShootingLayer(
        prob2_ad, Tsit5(),
        Tunable(1; kind = Val(:parameter), lowerbound = 10.0, upperbound = 170.0) => nothing,
    )
    linkage_ad = StageLinkage(
        switch_at_ts; u0 = u0_linkage, θ = [0.1], bounds_θ = ([0.0], [0.1])
    )
    chain_ad = Chain(stage1_ad, linkage_ad, stage2_ad)
    duration_constraint_ad = (ps -> [
        ps.layer_1.parameter.parameters.p₁.parameter.ps[1] +
            ps.layer_3.parameter.parameters.p₁.parameter.ps[1] - t_p
    ]) => (lb = [-Inf], ub = [0.0])
    return OptimizationProblem(
        chain_ad, AD, Val(:ComponentArrays);
        loss = :(-u₇),
        constraints = ((:u₄, 2) => (t = [1.0], bounds = (c_d_w, Inf)),),
        parameter_constraints = duration_constraint_ad,
    )
end

@testset "AD compatibility: $AD" for AD in (AutoForwardDiff(), AutoZygote())
    optprob_ad = make_reactor_op(AD)
    @test isapprox(optprob_ad.f(optprob_ad.u0, optprob_ad.p), -19.670789920209636, atol = 1.0e-6)
    sol_ad = solve(
        optprob_ad, Ipopt.Optimizer(); max_iter = 5,
        hessian_approximation = "limited-memory", print_level = 0,
    )
    @test isfinite(sol_ad.objective)
end
