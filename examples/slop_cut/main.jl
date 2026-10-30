#src ---
#src title: Optimization of the Slop-Cut Batch Distillation Problem
#src description: Multistage optimal control of a ternary batch distillation with production and slop cuts - grid-refinement workflow with consistent DAE initialization
#src tags:
#src   - Multi-stage
#src   - DAE
#src   - Optimal control
#src   - Batch distillation
#src   - Switching time
#src icon: ⚗️
#src ---

# ## Optimization of the Slop-Cut Batch Distillation Problem
#
# This tutorial **optimizes** the *slop-cut* batch distillation problem of Sager,
# *Numerical methods for mixed-integer optimal control problems*, PhD thesis (2005),
# which goes back to the MUSCOD-II batch distillation benchmark. It is the optimization
# counterpart of the simulation tutorial; the model equations are developed there and only
# summarized here.
#
# A ternary mixture is distilled in a column with five trays, a reboiler (the *pot*) and a
# total condenser. The batch is split into two *production cuts* ``P_1, P_2`` separated by
# two *slop cuts* ``S_1, S_2``. The manipulated variable is the **reflux ratio** ``R``, a
# piecewise-constant *staircase* with ``n`` nodes inside every cut; each cut also has a free
# duration ``T_k``. The objective is the profit
#
# ```math
# \Phi = \frac{4.5}{100}\,(P_1 + P_2 + P_3) - T ,
# ```
#
# where ``P_1, P_2`` are the masses of the two production cuts, ``P_3`` the final bottoms
# holdup and ``T`` the total batch time (Sager 2005, eq. 7.21, with unit energy cost and
# zero slop-cut disposal cost). The constraints are the purity specifications
#
# ```math
# x_{1,d}^{P_1} \ge 0.98, \qquad x_{2,d}^{P_2} \ge 0.96, \qquad x_2 + x_3 \le 0.01 .
# ```
#
# Two numerical properties of this problem dominate everything else and are the main lesson
# of this tutorial:
#
# 1. **The algebraic states must be re-initialized consistently whenever the reflux ratio
#    changes.** The VLE relations depend strongly on ``R``; restarting a segment with a
#    stale algebraic state makes the implicit integrator fail its first step
#    (`ReturnCode.Unstable`, `dt` underflow) — and a failed segment silently returns its
#    initial state, corrupting the trajectory **and zeroing the sensitivities** of
#    everything downstream.
# 2. **A single grid is not enough.** At the coarse control grid the discrete optimum is
#    visibly suboptimal; refining the grid around the coarse solution (grid refinement)
#    recovers the published optimum stably, while attacking the fine grid directly is
#    fragile.
#
# The workflow is therefore: simulate → optimize on a coarse grid → refine the grid,
# re-deriving consistent interface states at every step.

# ## Implementation

using Corleone
using Lux
using LuxCore
using Random
using ComponentArrays
using NonlinearSolve
using OrdinaryDiffEqBDF
using OrdinaryDiffEqNonlinearSolve: BrownFullBasicInit
using Optimization
using OptimizationMOI
using Ipopt
using CairoMakie

# Color per cut (P1, S1, P2, S2), used consistently in all plots.
const COLORS = [:steelblue, :gray45, :firebrick1, :gray35]

# ### Model constants
#
# Taken from the MUSCOD-II `slop.dat` data file. The Antoine coefficients are the entries
# selected by `CIDX01 = 0`, `CIDX02 = 1`, `CIDX03 = 6` on the 30-compound table in the
# MUSCOD-II C source (`slop.cpp`).

const NTRAYS = 5
const FEED0 = 100.0
const FEED1 = 0.33
const FEED2 = 0.33
const V0 = 110.0
const PRES = 1015.5

const A0 = [7.63846, 7.17331, 8.65385]
const B0 = [1976.3, 1779.1, 3149.1]
const C0 = [231.0, 186.0, 273.0]

keq(i, pres, temp) = 10.0^(A0[i] - B0[i] / (C0[i] + temp)) / pres

# ### Algebraic residual
#
# `slop_alg!` returns the 20 algebraic residuals ``g`` (vapour–liquid equilibrium on the pot,
# the five trays and the condenser) for a given pot state `xd`, reflux ratio `R` and pressure
# `Pres`. The index layout is `[pot x₃, pot T, (tray x₁,x₂,T)×5, condenser x₁,x₂,x₃]`.

function slop_alg!(g, X, xd, R, Pres)
    coe2 = 1.0 / (R + 1.0)
    coe1 = R * coe2
    g[1] = 1.0 - xd[2] - xd[3] - X[1]
    tp = X[2]
    y1 = keq(1, Pres, tp) * xd[2]
    y2 = keq(2, Pres, tp) * xd[3]
    y3 = keq(3, Pres, tp) * X[1]
    g[2] = 1.0 - y1 - y2 - y3
    g[3] = y1 - coe1 * X[3] - coe2 * X[18]
    g[4] = y2 - coe1 * X[4] - coe2 * X[19]
    for j in 1:NTRAYS
        bi = 4 + 3 * (j - 1)
        k = 3j
        x1j = X[k]
        x2j = X[k + 1]
        Tj = X[k + 2]
        z1 = keq(1, Pres, Tj) * x1j
        z2 = keq(2, Pres, Tj) * x2j
        z3 = keq(3, Pres, Tj) * (1.0 - x1j - x2j)
        nx1 = j < NTRAYS ? X[k + 3] : X[18]
        nx2 = j < NTRAYS ? X[k + 4] : X[19]
        g[bi + 1] = 1.0 - z1 - z2 - z3
        g[bi + 2] = z1 - coe1 * nx1 - coe2 * X[18]
        g[bi + 3] = z2 - coe1 * nx2 - coe2 * X[19]
    end
    g[20] = 1.0 - X[18] - X[19] - X[20]
    return
end

# ### Consistent initial conditions
#
# The start values in `slop.dat` are inconsistent, so the algebraic state is obtained once
# from a Newton solve. **Every** stage that starts at a changed reflux ratio needs this —
# see the DAE right-hand side below.

const X0GUESS = [
    0.34, 223.0,
    0.6228, 0.2859, 208.9,
    0.8293, 0.1565, 200.8,
    0.9296, 0.0686, 197.7,
    0.9726, 0.0272, 196.4,
    0.9899, 0.0100, 195.96,
    0.9968, 0.0032, 1.2e-6,
]

function consistent_xa(xd, R, Pres)
    f = NonlinearFunction((X, p) -> begin
        g = similar(X, 20)
        slop_alg!(g, X, xd, R, Pres)
        return g
    end)
    sol = solve(
        NonlinearProblem(f, Float64.(X0GUESS)), NewtonRaphson();
        abstol = 1.0e-12, reltol = 1.0e-12, maxiters = 200
    )
    return sol.u
end

# ### DAE right-hand side
#
# The state is 27-dimensional:
#
# * `u[1:3]` — pot holdup `M`, pot mole fractions `x1d, x2d`;
# * `u[4:23]` — the 20 algebraic states `X`;
# * `u[24]` — accumulated distillate `D`;
# * `u[25], u[26]` — accumulators `∫D·y₁`, `∫D·y₂` used for the product purities;
# * `u[27]` — accumulated physical time.
#
# Each cut lives on the normalized time ``\tau \in [0,1]`` and the duration `T` scales the
# source terms, so the physical time is recovered as ``\tau \mapsto \tau T`` and `u[27]`
# accumulates ``\int T\,\mathrm{d}\tau``. A type parameter `F` marks the slop cut `S2`, in
# which the pot fraction `x1d` is frozen.

const DV = [true, true, true, fill(false, 20)..., true, true, true, true]

struct SlopDAE{F} end

function (::SlopDAE{F})(res, du, u, p, t) where {F}
    Mv = u[1]
    x1d = u[2]
    x2d = u[3]
    X = @view u[4:23]
    R = p[1]
    T = p[2]
    V = p[3]
    Pres = p[4]
    dfr = V / (R + 1.0)
    res[1] = du[1] + T * dfr
    res[2] = du[2] + (F ? 0.0 : T * dfr * (X[18] - x1d) / Mv)
    res[3] = du[3] + T * dfr * (X[19] - x2d) / Mv
    g = @view res[4:23]
    slop_alg!(g, X, (Mv, x1d, x2d), R, Pres)
    res[24] = du[24] - T * dfr
    res[25] = du[25] - T * dfr * X[18]
    res[26] = du[26] - T * dfr * X[19]
    res[27] = du[27] - T
    return
end

function initstate(R)
    xd = [FEED0, FEED1, FEED2]
    X = consistent_xa(xd, R, PRES)
    u = vcat(xd, X, 0.0, 0.0, 0.0, 0.0)
    du = zeros(27)
    dfr = V0 / (R + 1)
    du[1] = -dfr
    du[2] = -dfr * (X[18] - xd[2]) / xd[1]
    du[3] = -dfr * (X[19] - xd[3]) / xd[1]
    return u, du
end

# ### Stages
#
# One cut is one `SingleShootingLayer`. The reflux ratio is a piecewise-constant control on
# the `NODES`-node grid, the cut duration is a free stage parameter, and the integration is
# pinned to a fixed-step implicit Euler (`adaptive=false, dt`); the fixed grid keeps the
# sensitivities of this stiff DAE well behaved.
#
# `dae_initialization = BrownFullBasicInit()` is the important switch: it makes the
# integrator *re-solve the algebraic states* at the start of **every** segment. Since a
# `SingleShootingLayer` splits its time span at every control breakpoint, the reflux ratio
# changes there — and with `NoInit` the carried algebraic state is then inconsistent for the
# new value. With `BrownFullBasicInit` every segment starts consistent; the cost is a small
# Newton solve per segment, which is cheap compared with a silently corrupted trajectory.

const SOLVE_KWARGS = (; adaptive = false, dt = 5.0e-4)

function mkstage(F, R0, T0; Tlb = 0.10, Tub = 1.70)
    grid = collect(range(0.0, 1.0; length = length(R0) + 1))[1:length(R0)]
    u, du = initstate(R0[1])
    prob = DAEProblem(
        SlopDAE{F}(), du, u, (0.0, 1.0), [R0[1], T0, V0, PRES];
        differential_vars = DV
    )
    return SingleShootingLayer(
        prob, DImplicitEuler();
        controls = (1 => ControlParameter(grid; name = :R, bounds = (0.3, 15.0),
            controls = collect(R0)),),
        params = [2],
        bounds_p = ([Tlb], [Tub]),
        dae_initialization = BrownFullBasicInit(),
        SOLVE_KWARGS...
    )
end

# ### Coupling the cuts — multiple shooting
#
# The cuts are coupled with `StageLinkage`: each linkage owns the next cut's interface state
# `u0` (a free, bounded variable) and the coupling `u0 - x_f(prev) = 0` is imposed as an
# equality constraint. The second linkage resets the distillate accumulators before the
# production cut `P2`. This keeps every DAE segment anchored to a controlled initial state
# and keeps the constraint Jacobian sparse; pure end-to-end single shooting over four
# coupled stiff DAE stages has ill-conditioned sensitivities and stalls Ipopt.

const NLINK = 27

link_carry() = StageLinkage(
    (x, θ) -> x[1:27]; u0 = zeros(NLINK),
    bounds_u0 = (fill(-1.0e3, NLINK), fill(1.0e3, NLINK))
)

function link_reset()
    f = (x, θ) -> begin
        y = copy(x[1:27]); y[24] = 0.0; y[25] = 0.0; y[26] = 0.0; y
    end
    return StageLinkage(f; u0 = zeros(NLINK),
        bounds_u0 = (fill(-1.0e3, NLINK), fill(1.0e3, NLINK)))
end

function build_chain(T0, R)
    s1 = mkstage(false, R[1], T0[1])
    s2 = mkstage(false, R[2], T0[2])
    s3 = mkstage(false, R[3], T0[3])
    s4 = mkstage(true, R[4], T0[4])
    return Chain(s1, link_carry(), s2, link_reset(), s3, link_carry(), s4)
end

# ### Objective and constraints
#
# The purities are imposed in their linearised form (`x25 - 0.98 x24 ≥ 0`,
# `x26 - 0.96 x24 ≥ 0`, `x2 + x3 ≤ 0.01`), and the pot holdup is kept above a small floor.
# The profit loss mirrors `Φ = 4.5/100 · (P1+P2+P3) - T` as a minimization:
# stage 1 contributes `-c·x₂₄` (product P1), stage 3 the same (product P2, accumulators
# reset), and stage 4 `x₂₇ - c·x₁` (total time minus product P3 = final holdup).

const P1QL = 0.98
const P2QL = 0.96
const P3QL = 0.99
const CPRICE = 4.5
const FEEDNM = 100.0
const M_MIN = 2.0
const CP = CPRICE / FEEDNM

function make_problem(chain)
    constraints = (
        (:(x₂₅ - 0.98 * x₂₄), 1) => (t = [1.0], bounds = (0.0, Inf)),
        (:(x₂₆ - 0.96 * x₂₄), 3) => (t = [1.0], bounds = (0.0, Inf)),
        (:(x₂ + x₃), 4) => (t = [1.0], bounds = (-Inf, 1.0 - P3QL)),
        (:x₁, 1) => (t = collect(0.1:0.3:1.0), bounds = (M_MIN, Inf)),
        (:x₁, 2) => (t = collect(0.1:0.3:1.0), bounds = (M_MIN, Inf)),
        (:x₁, 3) => (t = collect(0.1:0.3:1.0), bounds = (M_MIN, Inf)),
        (:x₁, 4) => (t = collect(0.0:0.2:1.0), bounds = (M_MIN, Inf)),
    )
    loss = [(:(-$(CP) * x₂₄), 1), (:(-$(CP) * x₂₄), 3), (:(x₂₇ - $(CP) * x₁), 4)]
    return OptimizationProblem(
        chain, AutoForwardDiff(), Val(:ComponentArrays);
        loss = loss, constraints = constraints
    )
end

# ### Consistent warm starts — the crucial helper
#
# Given candidate durations and staircase controls, this derives the three interface states
# by running a **pass-through chain** (`StageTransition` instead of `StageLinkage`) with the
# same controls, durations and integrator settings. The returned states are then exactly
# consistent with the linkage chain: the linkage residuals vanish (`≈ 1e-16`) at the Ipopt
# start point. Skipping this step — resampling the controls but reusing the old interface
# states — leaves residuals of order `1e-2`, and Ipopt burns its iteration budget restoring
# feasibility before it optimizes anything.
#
# The pre-pass must use the *same* `dae_initialization` as the optimization chain, otherwise
# the two trajectories differ and the residuals do not vanish.

function consistent_u0s(T0, R)
    pre = build_pass_through(T0, R)
    ps_pre, st_pre = LuxCore.setup(Random.default_rng(), pre)
    fpre = Corleone._forward_collect_stage_trajs(pre, ps_pre, st_pre)
    u1 = copy(last(fpre[1].u)[1:27])
    u2 = copy(last(fpre[2].u)[1:27]); u2[24] = 0.0; u2[25] = 0.0; u2[26] = 0.0
    u3 = copy(last(fpre[3].u)[1:27])
    return (u1, u2, u3)
end

function build_pass_through(T0, R)
    s1 = mkstage(false, R[1], T0[1])
    s2 = mkstage(false, R[2], T0[2])
    s3 = mkstage(false, R[3], T0[3])
    s4 = mkstage(true, R[4], T0[4])
    carry = StageTransition((traj, _) -> copy(last(traj.u)[1:27]))
    reset = StageTransition((traj, _) -> begin
        u = copy(last(traj.u)[1:27]); u[24] = 0.0; u[25] = 0.0; u[26] = 0.0; u
    end)
    return Chain(s1, carry, s2, reset, s3, carry, s4)
end

# ### Grid refinement
#
# The fine control grid is too ambitious as a *first* solve. Instead:
#
# 1. optimize on a coarse grid (4 nodes per cut) — few variables, robust;
# 2. resample the coarse solution's staircase onto the finer grid (piecewise constant);
# 3. re-derive the consistent interface states with `consistent_u0s`;
# 4. re-optimize; repeat.
#
# Each level warm-starts the next, and the warm start is *feasible and consistent*, so
# Ipopt starts at (nearly) the previous level's objective instead of fighting its way back
# to feasibility.

const LEVELS = [
    #(n = 4, iters =  20, dt = 1.0e-3),
    #(n = 10, iters = 20, dt = 1.0e-3),
    (n = 20, iters =  400, dt = 5.0e-4),
]

# Piecewise-constant resample of an n₀-node staircase onto n nodes (left-inverse of the
# midpoint sampling the integrator uses): node i of the coarse grid covers the same
# physical interval as a block of fine nodes.
resample_stair(R, n) = [R[clamp(Int(floor((i - 0.5) * length(R) / n)) + 1, 1, length(R))]
                        for i in 1:n]

# ### Solve

# Initial guess: a reasonable hand-built batch, P1–S1–P2–S2 with reflux staircase
# increasing within each cut (the optimal policy pushes `R` up as the pot depletes).
T0 = [1.1746618894595116, 0.5689127497722057, 1.1533048568406312, 0.6009589878645355]
R_init = [
    collect(range(1.4, 4.0; length = LEVELS[1].n)),
    collect(range(1.6, 5.0; length = LEVELS[1].n)),
    collect(range(6.0, 12.0; length = LEVELS[1].n)),
    collect(range(6.0, 12.0; length = LEVELS[1].n)),
]

# Every level **is** warm-started from the previous one: the loop keeps the current
# solution in `T, R`, `resample_stair` upsamples the previous level's staircase onto the
# next grid (identity on an unchanged grid), and `consistent_u0s` re-derives the interface
# states for the *upsampled* controls. Nothing else is carried over - once durations and
# controls are fixed, the remaining warm-start data is fully determined by the pass-through
# chain.
T, R = copy(T0), R_init
solutions = []
for lvl in LEVELS
    R = [resample_stair(R[k], lvl.n) for k in 1:4]
    chain = build_chain(T, R)
    u0s = consistent_u0s(T, R)
    # write the consistent interface states into the linkage variables
    for (i, lidx) in enumerate((2, 4, 6))
        getproperty(chain.layers, Symbol(:layer_, lidx)).u0 .= u0s[i]
    end
    ps, st = LuxCore.setup(Random.default_rng(), chain)
    x0 = ComponentArray(ps)

    sol = solve(
        make_problem(chain), Ipopt.Optimizer();
        hessian_approximation = "limited-memory",
        max_iter = lvl.iters, tol = 1.0e-6, constr_viol_tol = 1.0e-6,
        #mu_strategy = "monotone", bound_relax_factor = 1.0e-8,
        print_level = 5
    )
    xopt = sol + zero(x0)
    push!(solutions, (; lvl, chain, xopt, sol))
    println("n=$(lvl.n): retcode=", sol.retcode, " obj=", round(sol.objective; digits = 6))
    # next level starts from this solution
    T = [xopt.layer_1.p[1], xopt.layer_3.p[1], xopt.layer_5.p[1], xopt.layer_7.p[1]]
    R = [collect(xopt.layer_1.controls), collect(xopt.layer_3.controls),
        collect(xopt.layer_5.controls), collect(xopt.layer_7.controls)]
end

# ### Evaluate the final solution

final = solutions[end]
xopt = final.xopt
chain = final.chain
ps, st = LuxCore.setup(Random.default_rng(), chain)
trajs = Corleone._forward_collect_stage_trajs(chain, xopt, st)
e1, e2, e3, e4 = (last(trajs[i].u) for i in 1:4)
P1 = e1[24]
P2 = e3[24]
P3 = e4[1]
T = e4[27]
profit = CPRICE * (P1 + P2 + P3) / FEEDNM - T

println("profit = ", round(profit; digits = 5), "   T = ", round(T; digits = 5))
println("P1 = ", round(P1; digits = 4), "  P2 = ", round(P2; digits = 4),
    "  P3 = ", round(P3; digits = 4))
println("pur1 = ", round(e1[25] / e1[24]; digits = 6),
    "  pur2 = ", round(e3[26] / e3[24]; digits = 6),
    "  x3 = ", round(1 - e4[2] - e4[3]; digits = 6))

# ### Plot the optimal batch
#
# The reflux staircase is read back from the optimized parameters. For plotting, every cut
# is re-simulated **node by node** with `saveat` on the integrator grid: the trajectory
# stored by a `SingleShootingLayer` only contains the state at the control-node boundaries
# (`save_everystep = false`, one point per segment), which is far too coarse to plot.
# Re-running each node's DAE with the identical settings (`BrownFullBasicInit`, same `dt`)
# reproduces the node endpoint values exactly and yields the full step resolution in
# between. The re-simulation is cheap and is never part of the optimization.

opt_T = [xopt.layer_1.p[1], xopt.layer_3.p[1], xopt.layer_5.p[1], xopt.layer_7.p[1]]
opt_R = [collect(xopt.layer_1.controls), collect(xopt.layer_3.controls),
         collect(xopt.layer_5.controls), collect(xopt.layer_7.controls)]

plot_chain = build_pass_through(opt_T, opt_R)
ps_plot, st_plot = LuxCore.setup(Random.default_rng(), plot_chain)
fpre = Corleone._forward_collect_stage_trajs(plot_chain, ps_plot, st_plot)

# Dense re-simulation for plotting. `fpre[k].u` only contains the state at the
# control-node boundaries (`save_everystep = false` inside the layer), one point per
# segment - far too coarse for a plot. So every node's DAE is re-run on its physical
# sub-span with `saveat` on a fine output grid, starting from the node-boundary state of
# the layer trajectory. Three details matter:
#
# * the layer trajectory states are 28-long (the layer appends the active control value
#   as slot 28); the DAE state is the first 27 entries;
# * restarting from a carried state requires `du0 = zeros(27)` plus
#   `BrownFullBasicInit()` - the default consistency check rejects the carried algebraic
#   state at solver tolerance (`CheckInitFailureError`), while Brown's initialization
#   re-solves it and reproduces the layer trajectory;
# * node endpoints after re-initialization match the layer trajectory to solver
#   tolerance (~1e-7 here), not bitwise - hence the tolerance check.
#
# Note the node problem runs on the *normalized* span with the **unscaled** duration in
# `p[2]`: the DAE already multiplies every source term by `T`, so re-scaling `T` by the
# node fraction would shrink the drain rate by `1/n`.
plotUs, plotTs = [], []
tstarts = [0.0; cumsum(opt_T)[1:3]]

stages = [l for l in plot_chain.layers if l isa SingleShootingLayer]
for k in 1:4
    prob_k = stages[k].problem
    u0s = [copy(fpre[k].u[i][1:27]) for i in eachindex(fpre[k].u)]
    T_k = opt_T[k]
    Us, Ts = [copy(u0s[1])], [tstarts[k]]
    n = length(opt_R[k])
    for i in 1:n
        # normalized node span: t ∈ (0, 1/n), full duration in p[2]
        sol = solve(remake(prob_k; u0 = u0s[i], du0 = zeros(27),
                tspan = (0.0, 1 / n), p = [opt_R[k][i], T_k, V0, PRES]),
            DImplicitEuler(); adaptive = false, dt = SOLVE_KWARGS.dt,
            saveat = collect(range(0.0, 1 / n; length = 10 + 1)),
            initializealg = BrownFullBasicInit())
        maximum(abs, sol.u[end][1:27] - u0s[i + 1]) < 1e-6 ||
            error("resimulation disagrees with layer trajectory at stage $k node $i")
        append!(Us, [sol.u[j][1:27] for j in 2:length(sol.u)])
        append!(Ts, tstarts[k] .+ T_k .* (((i - 1) / n) .+
            collect(range(0.0, 1 / n; length = 10 + 1))[2:end]))
    end
    push!(plotUs, Us); push!(plotTs, Ts)
end

f = Figure(size = (1100, 750))
axR = CairoMakie.Axis(f[1, 1]; title = "Reflux ratio", ylabel = "R",
    yticks = 0:3:15, limits = (nothing, (-1, 16))
)
axM = CairoMakie.Axis(f[2, 1]; title = "Pot holdup", ylabel = "M", 
        yticks = 0:10:100,
        limits=(nothing, (-5, 105)), xlabel="Time t")

linkxaxes!(axR, axM)
hidexdecorations!(axR; grid = false)
#hidexdecorations!(axM; grid = false)

# time axis: physical time per stage, stitched at the stage boundaries
for k in 1:4
    tt, Us = plotTs[k], plotUs[k]

    # reflux staircase (one horizontal segment per node)
    tn = collect(range(0.0, 1.0; length = length(opt_R[k]) + 1))
    tc = tstarts[k] .+ opt_T[k] .* tn
    stairs!(axR, tc, vcat(opt_R[k], opt_R[k][end]); step = :post, color = COLORS[1])

    # holdup
    lines!(axM, tt, [u[1] for u in Us]; color = COLORS[2])
end

for ax in (axR, axM)
    for s in cumsum(opt_T)[1:3]
        vlines!(ax, s; linestyle = :dot, alpha = 0.4, color = :gray30)
    end
end
f
save(joinpath(@__DIR__, "slop_cut.pdf"), f)

# ## References
#
# * Sager, S. *Numerical methods for mixed-integer optimal control problems*. PhD thesis,
#   University of Heidelberg, 2005 (§7.3, slop-cut benchmark).
# * The MUSCOD-II benchmark description (`slop.dat`, `slop.cpp`).
