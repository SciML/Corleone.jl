#src ---
#src title: Slop-Cut Batch Distillation with Waste-Cut Recycling
#src description: Multistage optimal control of a ternary batch distillation where the slop cuts are recycled from the previous batch into the column
#src tags:
#src   - Multi-stage
#src   - DAE
#src   - Optimal control
#src   - Batch distillation
#src   - Recycling
#src icon: ♻️
#src ---

# ## Slop-Cut Batch Distillation with Waste-Cut Recycling
#
# This tutorial extends the *slop-cut* batch distillation problem of Sager,
# *Numerical methods for mixed-integer optimal control problems*, PhD thesis (2005), §7.3
# (the MUSCOD-II `slop` benchmark) to **waste-cut recycling**. The base problem — two
# production cuts ``P_1, P_2`` separated by two slop cuts ``S_1, S_2``, optimized over the
# reflux ratio ``R`` — is covered by the `slop_cut` example; only the recycle extension is
# developed here.
#
# ### The recycle idea
#
# In the base problem the slop cuts are waste: they are collected, set aside and the run
# stops once the pot is exhausted. In a *cyclically operated* batch plant the previous
# batch's slop cut **is** the feed of the next batch's first production period. Sager models
# this with two *slop reservoirs* ``\ell \in \{S_1, S_2\}`` that hold the accumulated slop
# of the previous cycle and are fed back into the column while the matching production cut
# runs:
#
# ```math
# \begin{aligned}
# \dot M   &= -D + \sum_{\ell} s_\ell , &
# \dot x_{1,d} &= -\frac{D}{M}(x_{1,d} - y_1) + \frac{1}{M}\sum_\ell s_\ell\,(x_{1,\ell} - x_{1,d}), \\
# \dot m_\ell &= - s_\ell , &
# \dot D   &= D , \qquad \dot I_i = D\,y_i ,
# \end{aligned}
# ```
#
# with ``s_\ell = \sum_{i\ge 1} f_{\ell,i}`` the total recycle flow of reservoir ``\ell``
# and ``f_{\ell,i}`` the flow routed to tray ``i`` (index `1` is the pot). The *recycle
# composition* ``x_{1,\ell}, x_{2,\ell}`` is a model parameter (it is the average
# composition of the slop cut, which is determined by the periodic closure, not by the
# current batch). Because every tray has constant holdup, the recycle mass added to a tray
# still increases the only variable holdup — the pot — at the total recycle rate, and the
# tray equations receive the corresponding composition advection terms
# (Sager 2005, eq. 7.26-7.28; `slopint.cpp`, `ffcn`/`gfcn`).
#
# The **recycling ratio** ``R_\ell \in [0,1]`` is the fraction of the previous slop cut that
# is fed back. Here it is encoded by the *reservoir initial holdup* ``m_\ell(0)`` together
# with the integrated recycle flow: ``R_\ell = m_\ell(0) - m_\ell(1)`` normalised by the
# previous cut. A full recycle (``R_\ell = 1``) empties the reservoir, no recycle
# (``R_\ell = 0``) starts it empty.
#
# ### Three variants in this file
#
# | variant | recycle | route |
# |:--|:--|:--|
# | **A** | none | base problem (reference) |
# | **B** | yes | every reservoir feeds the **pot** (single recycle stream) |
# | **C** | yes | every reservoir may feed **any tray** (flexible recycle) |
#
# Variants B and C correspond to Sager's scenarios B and C. The published optimum of
# scenario C is ``\Phi = 0.86312`` with ``T = 3.63688`` (`p4..p7` at the slop compositions,
# full recycle); with the valve routing frozen it is ``T = 3.64208,\ \Phi = 0.85792``.
# The recycle DAE used below is validated *exactly* against the MUSCOD-II reference
# trajectory `DAT/slopint.dat` in `context/slop-cut/recycle_proto.jl` (piecewise-linear
# reflux, all 29 states and the total time match to integration tolerance).

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

# ### Model constants
#
# Identical to the base example (`A0`,`B0`,`C0` are the Antoine entries selected by
# `CIDX01 = 0, CIDX02 = 1, CIDX03 = 6` in `slop.cpp`).

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

const X0GUESS = [
    0.34, 223.0,
    0.6228, 0.2859, 208.9,
    0.8293, 0.1565, 200.8,
    0.9296, 0.0686, 197.7,
    0.9726, 0.0272, 196.4,
    0.9899, 0.0100, 195.96,
    0.9968, 0.0032, 1.2e-6,
]

# ### Base algebraic residual (no recycle)
#
# The 20 algebraic residuals ``g`` (VLE on pot, the five trays and the condenser). Index
# layout: `[pot x₃, pot T, (tray x₁,x₂,T)×5, condenser x₁,x₂,x₃]`.

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

# ### Recycle algebraic residual
#
# Adds the recycle advection to the base residual. `f1`,`f2` are the six per-tray flows of
# reservoir 1 (``S_1``) and 2 (``S_2``); entry `i` is the flow to tray `i-1` (entry `1` is
# the pot). A flow routed to a tray contributes to that tray's VLE balance through the
# *sum of all flows at or below the destination* (`A1l`, `A2l`), because the added mass
# percolates down the column — see `ffcn`/`gfcn` and `rcfcn_1s` in `slopint.cpp`.

function slop_alg_recycle!(g, X, xd, R, Pres, p4, p5, p6, p7, f1, f2)
    slop_alg!(g, X, xd, R, Pres)
    A1 = 0.0
    A2 = 0.0
    for i in 2:6
        A1 += f1[i]
        A2 += f2[i]
    end
    g[3] += (A1 * (p4 - X[3]) + A2 * (p6 - X[3])) / V0
    g[4] += (A1 * (p5 - X[4]) + A2 * (p7 - X[4])) / V0
    for l in 1:NTRAYS
        bi = 4 + 3 * (l - 1)
        down = 3l + 3
        x1l = X[down]
        x2l = X[down + 1]
        A1l = 0.0
        A2l = 0.0
        for i in (l + 1):NTRAYS
            A1l += f1[i + 1]
            A2l += f2[i + 1]
        end
        g[bi + 2] += (A1l * (p4 - x1l) + A2l * (p6 - x1l)) / V0
        g[bi + 3] += (A1l * (p5 - x2l) + A2l * (p7 - x2l)) / V0
    end
    return
end

# ### Consistent initial conditions
#
# The `slop.dat` start values are inconsistent; the algebraic state is obtained from a
# Newton solve for the reflux ratio (and, in the recycle case, the recycle flows) at hand.

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

function consistent_xa_recycle(xd, R, Pres, p4, p5, p6, p7, f1, f2)
    f = NonlinearFunction((X, p) -> begin
        g = similar(X, 20)
        slop_alg_recycle!(g, X, xd, R, Pres, p4, p5, p6, p7, f1, f2)
        return g
    end)
    sol = solve(
        NonlinearProblem(f, Float64.(X0GUESS)), NewtonRaphson();
        abstol = 1.0e-12, reltol = 1.0e-12, maxiters = 200
    )
    return sol.u
end

# ### Base DAE (variant A)
#
# State (27): `[M, x1d, x2d, X(20), D, ∫D·y₁, ∫D·y₂, t]`. The type parameter `F` freezes
# the light pot fraction during the ``S_2`` cut (the reference does the same).

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
    res[2] = du[2] + (F ? 0.0 : T * dfr * (X[18] - x1d) / max(Mv, 0.5))
    res[3] = du[3] + T * dfr * (X[19] - x2d) / max(Mv, 0.5)
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
    du[24] = dfr
    du[25] = dfr * X[18]
    du[26] = dfr * X[19]
    du[27] = 1.0
    return u, du
end

# ### Recycle DAE (variants B and C)
#
# State (29): the base 27 plus the two reservoir holdups `m_{S1}, m_{S2}` appended.
# Parameters (20): `[R, T, V, Pres, p4, p5, p6, p7, f1(6), f2(6)]`, where `p4,p5` is the
# ``S_1`` recycle composition and `p6,p7` the ``S_2`` one. The reflux ratio is a
# piecewise-constant control (as in the base example); the recycle flows are either
# piecewise-constant controls (variant C) or fixed/controlled in a reduced way (variant B).

const DVR = [true, true, true, fill(false, 20)..., true, true, true, true, true, true]

struct RecycleDAE{F} end

function (::RecycleDAE{F})(res, du, u, p, t) where {F}
    Mv = u[1]
    x1d = u[2]
    x2d = u[3]
    X = @view u[4:23]
    R = p[1]
    T = p[2]
    V = p[3]
    Pres = p[4]
    p4 = p[5]
    p5 = p[6]
    p6 = p[7]
    p7 = p[8]
    f1 = @view p[9:14]
    f2 = @view p[15:20]
    s1 = sum(f1)
    s2 = sum(f2)
    dfr = V / (R + 1.0)
    Mf = max(Mv, 0.5)
    res[1] = du[1] + T * dfr - T * (s1 + s2)
    res[2] = du[2] + (F ? 0.0 : T * dfr * (X[18] - x1d) / Mf) -
             T * (s1 * (p4 - x1d) + s2 * (p6 - x1d)) / Mf
    res[3] = du[3] + T * dfr * (X[19] - x2d) / Mf -
             T * (s1 * (p5 - x2d) + s2 * (p7 - x2d)) / Mf
    g = @view res[4:23]
    slop_alg_recycle!(g, X, (Mv, x1d, x2d), R, Pres, p4, p5, p6, p7, f1, f2)
    res[24] = du[24] - T * dfr
    res[25] = du[25] - T * dfr * X[18]
    res[26] = du[26] - T * dfr * X[19]
    res[27] = du[27] - T
    res[28] = du[28] + T * s1
    res[29] = du[29] + T * s2
    return
end

function initstate_recycle(R, fl, x3_0, x4_0)
    xd = [FEED0, FEED1, FEED2]
    f1 = @view fl[1:6]
    f2 = @view fl[7:12]
    X = consistent_xa_recycle(xd, R, PRES, REC_P4, REC_P5, REC_P6, REC_P7, f1, f2)
    u = vcat(xd, X, 0.0, 0.0, 0.0, 0.0, x3_0, x4_0)
    du = zeros(29)
    dfr = V0 / (R + 1)
    s1 = sum(f1)
    s2 = sum(f2)
    du[1] = -dfr + (s1 + s2)
    du[2] = -dfr * (X[18] - xd[2]) / xd[1] +
            (s1 * (REC_P4 - xd[2]) + s2 * (REC_P6 - xd[2])) / xd[1]
    du[3] = -dfr * (X[19] - xd[3]) / xd[1] +
            (s1 * (REC_P5 - xd[3]) + s2 * (REC_P7 - xd[3])) / xd[1]
    du[24] = dfr
    du[25] = dfr * X[18]
    du[26] = dfr * X[19]
    du[27] = 1.0
    du[28] = -s1
    du[29] = -s2
    return u, du
end

# ### Scenario-C reference data
#
# The slop-cut compositions and durations of the published scenario-C optimum. They seed the
# recycle compositions (which are *held fixed* here — determining them is the periodic
# closure, see the closing remarks) and the initial reservoir holdups.

const REC_P4 = 0.8368715820055209   # S1 recycle x1
const REC_P5 = 0.16292259682826676  # S1 recycle x2
const REC_P6 = 3.83413877087826e-5  # S2 recycle x1
const REC_P7 = 0.7600396155394682   # S2 recycle x2

# Explanation follows after the model.

# ### Variants
#
# A `VariantSpec` only carries the routing structure; `recycle = false` selects the base DAE
# (variant A).

struct VariantSpec
    name::Symbol
    recycle::Bool
    flexible::Bool
end

const VARIANT_A = VariantSpec(:A, false, false)
const VARIANT_B = VariantSpec(:B, true, false)
const VARIANT_C = VariantSpec(:C, true, true)

# ### Stage construction
#
# One cut is one `SingleShootingLayer`. The reflux ratio is a piecewise-constant control on
# an `N`-node grid; the cut duration `T` is a free stage parameter. For the recycle variants
# the per-tray recycle flows `f` are additional piecewise-constant controls on the same grid.
# The base example's lesson still applies: `BrownFullBasicInit()` re-solves the algebraic
# states at **every** control breakpoint, where the reflux ratio (and the recycle flow)
# changes.

const SOLVE_KWARGS = (; adaptive = false, dt = 5.0e-4)

function _control_grid(n)
    return collect(range(0.0, 1.0; length = n + 1))[1:n]
end

function mkstage_base(which, R0, T0; Tlb = 0.10, Tub = 1.70)
    F = which === :S2
    u, du = initstate(R0[1])
    prob = DAEProblem(
        SlopDAE{F}(), du, u, (0.0, 1.0), [R0[1], T0, V0, PRES];
        differential_vars = DV
    )
    return SingleShootingLayer(
        prob, DImplicitEuler();
        controls = (1 => ControlParameter(_control_grid(length(R0)); name = :R,
            bounds = (0.3, 15.0), controls = collect(R0)),),
        params = [2],
        bounds_p = ([Tlb], [Tub]),
        dae_initialization = BrownFullBasicInit(),
        SOLVE_KWARGS...
    )
end

# In the recycle stage the parameter vector is
# `[R, T, V, Pres, p4, p5, p6, p7, f1₁..f1₆, f2₁..f2₆]`. The active recycle flows are
# selected by `which` and `flexible`:
#
# * `P1` drains reservoir 1 (the ``S_1`` recycle), controlled through `p[9:14]`;
# * `P2` drains reservoir 2 (the ``S_2`` recycle), controlled through `p[15:20]`;
# * `S1`/`S2` collect, so no flow is active.
#
# Every inactive flow slot is *fixed* at zero by leaving it out of `params`.

function mkstage_recycle(var, which, R0, T0, fl0; x3_0 = 14.44049965, x4_0 = 16.74616895,
        Tlb = 0.10, Tub = 1.70)
    F = which === :S2
    u, du = initstate_recycle(R0[1], fl0, x3_0, x4_0)
    p0 = [R0[1], T0, V0, PRES, REC_P4, REC_P5, REC_P6, REC_P7, collect(fl0)...]
    prob = DAEProblem(
        RecycleDAE{F}(), du, u, (0.0, 1.0), p0;
        differential_vars = DVR
    )
    grid = _control_grid(length(R0))
    ctrls = Any[1 => ControlParameter(grid; name = :R, bounds = (0.3, 15.0),
        controls = collect(R0))]
    flow_bounds = (0.0, 30.0)
    if which === :P1
        js = var.flexible ? (9:14) : (9:9)
        for j in js
            push!(ctrls, j => ControlParameter(grid; name = Symbol(:f1_, j - 8),
                bounds = flow_bounds, controls = fill(fl0[j - 8], length(grid))))
        end
    elseif which === :P2
        js = var.flexible ? (15:20) : (15:15)
        for j in js
            push!(ctrls, j => ControlParameter(grid; name = Symbol(:f2_, j - 14),
                bounds = flow_bounds, controls = fill(fl0[j - 8], length(grid))))
        end
    end
    ic_kw = if which === :P1
        lb_ic = fill(-1.0e3, NLINKR)
        ub_ic = fill(1.0e3, NLINKR)
        lb_ic[28] = 0.0
        ub_ic[28] = 25.0
        lb_ic[29] = 0.0
        ub_ic[29] = 25.0
        (; tunable_ic = [28, 29], bounds_ic = (lb_ic, ub_ic))
    else
        (;)
    end
    return SingleShootingLayer(
        prob, DImplicitEuler();
        controls = Tuple(ctrls),
        params = [2],
        bounds_p = ([Tlb], [Tub]),
        ic_kw...,
        dae_initialization = BrownFullBasicInit(),
        SOLVE_KWARGS...
    )
end

# ### Coupling the cuts
#
# `StageLinkage`s own the interface states; the second one (before ``P_2``) resets the
# distillate accumulators but **keeps** the reservoir holdups (they are drained during
# ``P_2``).

const NLINK = 27
const NLINKR = 29

link_carry() = StageLinkage(
    (x, θ) -> x[1:NLINK]; u0 = zeros(NLINK),
    bounds_u0 = (fill(-1.0e3, NLINK), fill(1.0e3, NLINK))
)

function link_reset()
    f = (x, θ) -> begin
        y = copy(x[1:NLINK])
        y[24] = 0.0
        y[25] = 0.0
        y[26] = 0.0
        y
    end
    return StageLinkage(f; u0 = zeros(NLINK),
        bounds_u0 = (fill(-1.0e3, NLINK), fill(1.0e3, NLINK)))
end

link_carry_r() = StageLinkage(
    (x, θ) -> x[1:NLINKR]; u0 = zeros(NLINKR),
    bounds_u0 = (fill(-1.0e3, NLINKR), fill(1.0e3, NLINKR))
)

function link_reset_r()
    f = (x, θ) -> begin
        y = copy(x[1:NLINKR])
        y[24] = 0.0
        y[25] = 0.0
        y[26] = 0.0
        y
    end
    return StageLinkage(f; u0 = zeros(NLINKR),
        bounds_u0 = (fill(-1.0e3, NLINKR), fill(1.0e3, NLINKR)))
end

# `flows(spec, which, amount, T, R)` builds the initial 12-vector of recycle flows. The pot
# variants put the whole stream on the pot (slot 1); the flexible variant spreads it evenly
# over the six trays (an interior-point start).

function flows0(var, which, amount, T, n)
    fl = zeros(12)
    idx = which === :P1 ? 1 : 7
    if amount <= 0
        return fl
    end
    rate = amount / max(T, 1.0e-3)
    if var.flexible
        fl[idx:(idx + 5)] .= rate / 6
    else
        fl[idx] = rate
    end
    return fl
end

function build_chain(var::VariantSpec, T0, R, F1, F2, x3_0, x4_0)
    if !var.recycle
        s1 = mkstage_base(:P1, R[1], T0[1])
        s2 = mkstage_base(:S1, R[2], T0[2])
        s3 = mkstage_base(:P2, R[3], T0[3])
        s4 = mkstage_base(:S2, R[4], T0[4])
        return Chain(s1, link_carry(), s2, link_reset(), s3, link_carry(), s4)
    end
    z12 = zeros(12)
    s1 = mkstage_recycle(var, :P1, R[1], T0[1], vcat(F1, zeros(6)); x3_0, x4_0)
    s2 = mkstage_recycle(var, :S1, R[2], T0[2], z12; x3_0, x4_0)
    s3 = mkstage_recycle(var, :P2, R[3], T0[3], vcat(zeros(6), F2); x3_0, x4_0)
    s4 = mkstage_recycle(var, :S2, R[4], T0[4], z12; x3_0, x4_0)
    return Chain(s1, link_carry_r(), s2, link_reset_r(), s3, link_carry_r(), s4)
end

# ### Consistent warm starts
#
# As in the base example, the interface states are re-derived by a pass-through chain
# (`StageTransition` instead of `StageLinkage`) with the *same* controls, durations and
# `BrownFullBasicInit`, so the linkage residuals vanish at the Ipopt start point.

function build_pass_through(var::VariantSpec, T0, R, F1, F2, x3_0, x4_0)
    if !var.recycle
        s1 = mkstage_base(:P1, R[1], T0[1])
        s2 = mkstage_base(:S1, R[2], T0[2])
        s3 = mkstage_base(:P2, R[3], T0[3])
        s4 = mkstage_base(:S2, R[4], T0[4])
        carry = StageTransition((traj, _) -> copy(last(traj.u)[1:NLINK]))
        reset = StageTransition((traj, _) -> begin
            u = copy(last(traj.u)[1:NLINK])
            u[24] = 0.0; u[25] = 0.0; u[26] = 0.0; u
        end)
        return Chain(s1, carry, s2, reset, s3, carry, s4)
    end
    z12 = zeros(12)
    s1 = mkstage_recycle(var, :P1, R[1], T0[1], vcat(F1, zeros(6)); x3_0, x4_0)
    s2 = mkstage_recycle(var, :S1, R[2], T0[2], z12; x3_0, x4_0)
    s3 = mkstage_recycle(var, :P2, R[3], T0[3], vcat(zeros(6), F2); x3_0, x4_0)
    s4 = mkstage_recycle(var, :S2, R[4], T0[4], z12; x3_0, x4_0)
    carry = StageTransition((traj, _) -> copy(last(traj.u)[1:NLINKR]))
    reset = StageTransition((traj, _) -> begin
        u = copy(last(traj.u)[1:NLINKR])
        u[24] = 0.0; u[25] = 0.0; u[26] = 0.0; u
    end)
    return Chain(s1, carry, s2, reset, s3, carry, s4)
end

function consistent_u0s(var::VariantSpec, T0, R, F1, F2, x3_0, x4_0)
    pre = build_pass_through(var, T0, R, F1, F2, x3_0, x4_0)
    ps_pre, st_pre = LuxCore.setup(Random.default_rng(), pre)
    fpre = Corleone._forward_collect_stage_trajs(pre, ps_pre, st_pre)
    if !var.recycle
        u1 = copy(last(fpre[1].u)[1:NLINK])
        u2 = copy(last(fpre[2].u)[1:NLINK]); u2[24] = 0.0; u2[25] = 0.0; u2[26] = 0.0
        u3 = copy(last(fpre[3].u)[1:NLINK])
        return (u1, u2, u3)
    end
    u1 = copy(last(fpre[1].u)[1:NLINKR])
    u2 = copy(last(fpre[2].u)[1:NLINKR]); u2[24] = 0.0; u2[25] = 0.0; u2[26] = 0.0
    u3 = copy(last(fpre[3].u)[1:NLINKR])
    return (u1, u2, u3)
end

# ### Objective and constraints
#
# The profit ``\Phi = \frac{4.5}{100}(P_1 + P_2 + P_3) - T`` is minimized as in the base
# example; recycling raises the products by re-using the slop holdup. The purity
# specifications and the pot-holdup floor are unchanged. For the recycle variants with a
# **pot-only** route the reservoir must empty by the end of the matching production cut
# (recycling ratio ``R_\ell = 1``); the flexible route leaves the residual holdup free
# (partial recycling).

const P1QL = 0.98
const P2QL = 0.96
const P3QL = 0.99
const CPRICE = 4.5
const FEEDNM = 100.0
const M_MIN = 2.0
const CP = CPRICE / FEEDNM

function make_problem(chain, var::VariantSpec; kwargs...)
    cs = Any[
        (:(x₂₅ - 0.98 * x₂₄), 1) => (t = [1.0], bounds = (0.0, Inf)),
        (:(x₂₆ - 0.96 * x₂₄), 3) => (t = [1.0], bounds = (0.0, Inf)),
        (:(x₂ + x₃), 4) => (t = [1.0], bounds = (-Inf, 1.0 - P3QL)),
        (:x₁, 1) => (t = collect(0.1:0.3:1.0), bounds = (M_MIN, Inf)),
        (:x₁, 2) => (t = collect(0.1:0.3:1.0), bounds = (M_MIN, Inf)),
        (:x₁, 3) => (t = collect(0.1:0.3:1.0), bounds = (M_MIN, Inf)),
        (:x₁, 4) => (t = collect(0.0:0.2:1.0), bounds = (M_MIN, Inf)),
    ]
    if var.recycle && !var.flexible
        # fully recycle the pot-only stream: both reservoirs empty by the end of P1 / P2
        push!(cs, (:(x₂₈ - 1.0e-3), 1) => (t = [1.0], bounds = (-Inf, 0.0)))
        push!(cs, (:(x₂₉ - 1.0e-3), 3) => (t = [1.0], bounds = (-Inf, 0.0)))
    end
    constraints = Tuple(cs)
    loss = [(:(-$(CP) * x₂₄), 1), (:(-$(CP) * x₂₄), 3), (:(x₂₇ - $(CP) * x₁), 4)]
    return OptimizationProblem(
        chain, AutoForwardDiff(), Val(:ComponentArrays);
        loss = loss, constraints = constraints, kwargs...
    )
end

# ### Initial guess
#
# The production/slop durations and the reflux ramps are taken from the scenario-C reference;
# the reservoir holdups start at the previous slop cuts (14.44 for ``S_1`` and 16.75 for
# ``S_2``), i.e. at full recycle.

const NODES = parse(Int, get(ENV, "SLOP_NODES", "4"))
const MAXITER = parse(Int, get(ENV, "SLOP_MAXITER", "60"))
const PRINTLEVEL = parse(Int, get(ENV, "SLOP_PRINT", "0"))

const T0REF = [1.1746618894595116, 0.5689127497722057, 1.1533048568406312, 0.6009589878645355]

# A good base-problem solution (`context/slop-cut/conv_fixed_20.txt`, profit 0.4444) is used
# as the start point for every variant. On the 20-node grid it is the identity; on a coarser
# grid it is resampled piecewise-constantly. Starting near the base optimum keeps Ipopt from
# wandering off while it adapts the production policy to the recycle stream.
const WARM_R = [
    [1.9846607626937565, 2.066618063165247, 2.162475728224359, 2.2629772985920544,
        2.3729591231704443, 2.494973660675954, 2.633329024450643, 2.7878756361397925,
        2.9706632945146194, 3.182776626699096, 3.40840172262416, 3.678796132623472,
        3.9903543694989114, 4.358418816019719, 4.791711007723939, 5.295420809026092,
        5.876980565882866, 6.582784105319068, 7.3891521050293125, 8.359506153369228],
    [4.7566032736599055, 5.189214766476991, 5.655463227221163, 6.160421862043851,
        6.729835072738008, 7.340605016624293, 7.970978029300608, 8.64747497604943,
        9.39982335052382, 10.226673080801822, 11.09981469194009, 11.983421863693035,
        12.845607113509155, 13.651762042298722, 14.376304865306642, 14.560020525458722,
        14.382217213021198, 14.724167840997012, 14.76283253753555, 14.821228741546511],
    [2.0514189853286537, 2.1691769328886115, 2.2873440895880006, 2.415738710547561,
        2.5485592600499243, 2.6969101567794986, 2.8699153319685897, 3.055456845330236,
        3.264039313974951, 3.493957160238856, 3.761954940275398, 4.066016406433276,
        4.415762052773476, 4.819892220362233, 5.2879185640586295, 5.831871194855493,
        6.467366394850378, 7.205686834494723, 8.067558391419697, 9.078156554710258],
    [5.02783550326289, 5.47579406392689, 5.972261730863942, 6.511721724974596,
        7.0850159428423645, 7.710543362062023, 8.372977042919246, 9.063880339590037,
        9.801046884981956, 10.597928396870117, 11.441368599825571, 12.30609674611192,
        13.162191247015649, 13.964194405113368, 14.574697870900104, 14.572515060431787,
        14.70199623396116, 14.765765879437563, 14.802310305946753, 14.826692062185927],
]

function resample_staircase(v, n)
    t_new = _control_grid(n)
    t_old = _control_grid(length(v))
    return [v[clamp(searchsortedlast(t_old, t), 1, length(v))] for t in t_new]
end

function initial_reflux(var::VariantSpec, n)
    if !var.recycle
        return [resample_staircase(WARM_R[k], n) for k in 1:4]
    end
    # scenario-C shape: gentle production ramps, steep slop ramps to the R = 15 ceiling
    return [
        collect(range(1.40, 2.35; length = n)),
        collect(range(2.30, 15.0; length = n)),
        collect(range(1.42, 2.73; length = n)),
        collect(range(2.10, 15.0; length = n)),
    ]
end

# ### Solve driver
#
# `run_variant` builds the chain, writes the consistent interface states and the initial
# reservoir holdups into the parameter vector, and solves with Ipopt (L-BFGS Hessian, as the
# base example). The knob `idx` in `layer_1.p` / `layer_3.p` / `layer_5.p` / `layer_7.p` is
# the cut duration `T`; the controls are read back from the layers.

function run_variant(var::VariantSpec; n = NODES, maxiter = MAXITER, print_level = 0)
    T0 = copy(T0REF)
    R = initial_reflux(var, n)
    F1 = flows0(var, :P1, 14.44049965, T0[1], n)
    F2 = flows0(var, :P2, 16.74616895, T0[3], n)
    x3_0 = var.recycle ? 14.44049965 : 0.0
    x4_0 = var.recycle ? 16.74616895 : 0.0

    chain = build_chain(var, T0, R, F1, F2, x3_0, x4_0)
    u0s = consistent_u0s(var, T0, R, F1, F2, x3_0, x4_0)
    for (i, lidx) in enumerate((2, 4, 6))
        getproperty(chain.layers, Symbol(:layer_, lidx)).u0 .= u0s[i]
    end
    ps, st = LuxCore.setup(Random.default_rng(), chain)
    x0 = ComponentArray(ps)
    if var.recycle
        x0.layer_1.u0 .= [x3_0, x4_0]
    end
    @assert (var.recycle ? length(x0.layer_1.u0) == 2 : length(x0.layer_1.u0) == 0)

    if maxiter < 0
        # diagnostic: evaluate the warm start without optimizing
        sol = nothing
        xopt = copy(x0)
    else
        sol = solve(
            make_problem(chain, var), Ipopt.Optimizer();
            hessian_approximation = "limited-memory",
            max_iter = maxiter,
            tol = 1.0e-6,
            constr_viol_tol = 1.0e-6,
            print_level = print_level
        )
        xopt = sol + zero(x0)
    end

    # evaluate the optimal batch
    ps_e, st_e = LuxCore.setup(Random.default_rng(), chain)
    trajs = Corleone._forward_collect_stage_trajs(chain, xopt, st_e)
    e1, e2, e3, e4 = (last(trajs[i].u) for i in 1:4)
    P1 = e1[24]
    P2 = e3[24]
    P3 = e4[1]
    T = e4[27]
    profit = CPRICE * (P1 + P2 + P3) / FEEDNM - T
    m1_0 = var.recycle ? xopt.layer_1.u0[1] : 0.0
    m2_0 = var.recycle ? xopt.layer_1.u0[2] : 0.0
    recycled = var.recycle ? (m1_0 - e4[28]) + (m2_0 - e4[29]) : 0.0
    pur1 = e1[25] / max(e1[24], 1.0e-12)
    pur2 = e3[26] / max(e3[24], 1.0e-12)
    pur3 = e4[2] + e4[3]
    return (;
        var, chain, xopt, sol,
        T = T, P1 = P1, P2 = P2, P3 = P3, profit = profit, recycled = recycled,
        pur1 = pur1, pur2 = pur2, pur3 = pur3,
        x3_0 = x3_0, x4_0 = x4_0, m1_0 = m1_0, m2_0 = m2_0,
        opt_T = [xopt.layer_1.p[1], xopt.layer_3.p[1], xopt.layer_5.p[1], xopt.layer_7.p[1]],
        opt_R = [collect(xopt.layer_1.controls), collect(xopt.layer_3.controls),
            collect(xopt.layer_5.controls), collect(xopt.layer_7.controls)],
    )
end

# ### Dense re-evaluation for plotting
#
# The shooting layer stores one point per control node, which is too coarse for the
# composition curves. Once the optimal staircase `R*`, durations `T*` and (for the recycle
# variants) flows `f*` are known, they are resampled onto a fine grid and the batch is
# re-integrated as a pass-through chain, giving a smooth trajectory for every state.

function resample_ctrl(v, n)
    n0 = length(v)
    told = _control_grid(n0)
    return [v[clamp(searchsortedlast(told, t), 1, n0)] for t in _control_grid(n)]
end

function mkstage_dense(var, which, Rv, Tv, f1v, f2v, x3_0, x4_0)
    grid = _control_grid(length(Rv))
    if !var.recycle
        u, du = initstate(Rv[1])
        prob = DAEProblem(
            SlopDAE{which === :S2}(), du, u, (0.0, 1.0), [Rv[1], Tv, V0, PRES];
            differential_vars = DV
        )
        ctrl = 1 => ControlParameter(grid; name = :R, bounds = (0.3, 15.0), controls = collect(Rv))
        return SingleShootingLayer(
            prob, DImplicitEuler(); controls = (ctrl,), params = [2],
            bounds_p = ([Tv], [Tv]), dae_initialization = BrownFullBasicInit(), SOLVE_KWARGS...
        )
    end
    fl0 = vcat([f1v[i][1] for i in 1:6], [f2v[i][1] for i in 1:6])
    u, du = initstate_recycle(Rv[1], fl0, x3_0, x4_0)
    p0 = [Rv[1], Tv, V0, PRES, REC_P4, REC_P5, REC_P6, REC_P7, fl0...]
    prob = DAEProblem(
        RecycleDAE{which === :S2}(), du, u, (0.0, 1.0), p0; differential_vars = DVR
    )
    ctrls = Any[1 => ControlParameter(grid; name = :R, bounds = (0.3, 15.0), controls = collect(Rv))]
    if which === :P1
        for i in 1:6
            any(!iszero, f1v[i]) || continue
            push!(ctrls, (8 + i) => ControlParameter(grid; name = Symbol(:f1_, i),
                bounds = (0.0, 30.0), controls = collect(f1v[i])))
        end
    elseif which === :P2
        for i in 1:6
            any(!iszero, f2v[i]) || continue
            push!(ctrls, (14 + i) => ControlParameter(grid; name = Symbol(:f2_, i),
                bounds = (0.0, 30.0), controls = collect(f2v[i])))
        end
    end
    return SingleShootingLayer(
        prob, DImplicitEuler(); controls = Tuple(ctrls), params = [2],
        bounds_p = ([Tv], [Tv]), dae_initialization = BrownFullBasicInit(), SOLVE_KWARGS...
    )
end

function build_dense_chain(var, Tv, Rf, F1f, F2f, x3_0, x4_0, nl)
    z6 = [zeros(nl) for _ in 1:6]
    if !var.recycle
        s1 = mkstage_dense(var, :P1, Rf[1], Tv[1], z6, z6, x3_0, x4_0)
        s2 = mkstage_dense(var, :S1, Rf[2], Tv[2], z6, z6, x3_0, x4_0)
        s3 = mkstage_dense(var, :P2, Rf[3], Tv[3], z6, z6, x3_0, x4_0)
        s4 = mkstage_dense(var, :S2, Rf[4], Tv[4], z6, z6, x3_0, x4_0)
        carry = StageTransition((traj, _) -> copy(last(traj.u)[1:NLINK]))
        reset = StageTransition((traj, _) -> begin
            u = copy(last(traj.u)[1:NLINK]); u[24] = 0.0; u[25] = 0.0; u[26] = 0.0; u
        end)
        return Chain(s1, carry, s2, reset, s3, carry, s4)
    end
    s1 = mkstage_dense(var, :P1, Rf[1], Tv[1], F1f, z6, x3_0, x4_0)
    s2 = mkstage_dense(var, :S1, Rf[2], Tv[2], z6, z6, x3_0, x4_0)
    s3 = mkstage_dense(var, :P2, Rf[3], Tv[3], z6, F2f, x3_0, x4_0)
    s4 = mkstage_dense(var, :S2, Rf[4], Tv[4], z6, z6, x3_0, x4_0)
    carry = StageTransition((traj, _) -> copy(last(traj.u)[1:NLINKR]))
    reset = StageTransition((traj, _) -> begin
        u = copy(last(traj.u)[1:NLINKR]); u[24] = 0.0; u[25] = 0.0; u[26] = 0.0; u
    end)
    return Chain(s1, carry, s2, reset, s3, carry, s4)
end

# Recover the per-tray recycle flows from the flat control vector of a stage layer: the
# first `n` entries are `R`, the remaining blocks of `n` are the active flow slots.
function flow_blocks(r, layer, n)
    c = collect(getproperty(r.xopt, layer).controls)
    nc = length(c) ÷ n - 1
    F = [zeros(n) for _ in 1:6]
    for i in 1:nc
        F[i] = c[(i * n + 1):((i + 1) * n)]
    end
    return F
end

function dense_trajs(r; nfine = 160)
    var = r.var
    n = length(r.opt_R[1])
    Rf = [resample_ctrl(r.opt_R[k], nfine) for k in 1:4]
    z6 = [zeros(nfine) for _ in 1:6]
    F1f = var.recycle ? [resample_ctrl(F, nfine) for F in flow_blocks(r, :layer_1, n)] : z6
    F2f = var.recycle ? [resample_ctrl(F, nfine) for F in flow_blocks(r, :layer_5, n)] : z6
    chain = build_dense_chain(var, r.opt_T, Rf, F1f, F2f, r.m1_0, r.m2_0, nfine)
    ps, st = LuxCore.setup(Random.default_rng(), chain)
    trajs = Corleone._forward_collect_stage_trajs(chain, ps, st)
    t = Float64[]
    cols = Vector{Float64}[]
    t0 = 0.0
    for k in 1:4
        tr = trajs[k]
        npts = length(tr.u)
        append!(t, t0 .+ r.opt_T[k] .* range(0.0, 1.0; length = npts))
        for u in tr.u
            push!(cols, collect(u))
        end
        t0 += r.opt_T[k]
    end
    return t, reduce(hcat, cols)
end

# ### Run the variants
#
# `SLOP_VARIANTS` selects a subset, e.g. `SLOP_VARIANTS=AC` solves variants A and C only.

const ALL_VARIANTS = Dict('A' => VARIANT_A, 'B' => VARIANT_B, 'C' => VARIANT_C)
const SELECTED = get(ENV, "SLOP_VARIANTS", "ABC")

results = VariantSpec[ALL_VARIANTS[c] for c in SELECTED if haskey(ALL_VARIANTS, c)]
solutions = map(results) do var
    println("── variant $(var.name) ──")
    r = run_variant(var; print_level = PRINTLEVEL)
    rc = isnothing(r.sol) ? :warmstart : r.sol.retcode
    println("retcode = ", rc,
        "   profit = ", round(r.profit; digits = 5),
        "   T = ", round(r.T; digits = 5))
    println("P1 = ", round(r.P1; digits = 4), "  P2 = ", round(r.P2; digits = 4),
        "  P3 = ", round(r.P3; digits = 4),
        "  recycled = ", round(r.recycled; digits = 4))
    println("pur1 = ", round(r.pur1; digits = 5), " (≥ 0.98)   pur2 = ",
        round(r.pur2; digits = 5), " (≥ 0.96)   light+mid = ", round(r.pur3; digits = 5),
        " (≤ 0.01)")
    flush(stdout)
    r
end

# ### Reference comparison
#
# The published scenario-C optimum is ``\Phi = 0.86312`` at ``T = 3.63688`` with a full
# recycle of both slop cuts. Because the recycle compositions `p4..p7` are held fixed and
# the periodic closure (recycled amount = slop produced *this* batch) is relaxed to a free
# reservoir initial holdup, the numbers below are expected to be in the right neighbourhood
# rather than identical; the *base* variant A is the exact published scenario-A problem and
# should reproduce ``\Phi \approx 0.445``.

println()
println("reference: A  Φ ≈ 0.445, T ≈ 3.500")
println("           B  Φ ≈ 0.760, T ≈ 3.740")
println("           C  Φ = 0.86312, T = 3.63688")

# ### Plot
#
# Every variant is re-integrated on a fine grid with its optimal controls, so the curves
# are smooth. Columns: reflux-ratio staircase, pot holdup `M`, pot composition
# ``x_{1d}, x_{2d}`` and instantaneous distillate (condenser) composition ``x_1, x_2``.
# Dotted vertical lines mark the cut boundaries ``P_1 | S_1 | P_2 | S_2``.

dens = [dense_trajs(r) for r in solutions]

f = Figure(size = (1600, 1000))
line_colors = [:steelblue, :gray45, :firebrick1]

for (vi, (r, (t, U))) in enumerate(zip(solutions, dens))
    edges = cumsum(r.opt_T)
    tstarts = [0.0; edges[1:3]]

    axR = CairoMakie.Axis(
        f[vi, 1]; title = "$(r.var.name): reflux ratio", ylabel = "R",
        yticks = 0:3:15, limits = (nothing, (-1, 16))
    )
    axM = CairoMakie.Axis(
        f[vi, 2]; title = "$(r.var.name): pot holdup", ylabel = "M",
        yticks = 0:20:100, limits = (nothing, (-5, 105))
    )
    axC = CairoMakie.Axis(
        f[vi, 3]; title = "$(r.var.name): pot composition", xlabel = "t", ylabel = "x",
        limits = (nothing, (-0.05, 1.05))
    )
    axD = CairoMakie.Axis(
        f[vi, 4]; title = "$(r.var.name): distillate composition", xlabel = "t",
        ylabel = "x", limits = (nothing, (-0.05, 1.05))
    )

    for k in 1:4
        n = length(r.opt_R[k])
        tc = tstarts[k] .+ r.opt_T[k] .* collect(range(0.0, 1.0; length = n + 1))
        stairs!(axR, tc, vcat(r.opt_R[k], r.opt_R[k][end]); step = :post,
            color = line_colors[vi])
    end

    lines!(axM, t, U[1, :]; color = line_colors[vi])
    lines!(axC, t, U[2, :]; color = :steelblue, label = "x₁ (pot)")
    lines!(axC, t, U[3, :]; color = :orange, label = "x₂ (pot)")
    lines!(axD, t, U[21, :]; color = :steelblue, label = "x₁ (dist)")
    lines!(axD, t, U[22, :]; color = :orange, label = "x₂ (dist)")

    for e in edges
        for ax in (axR, axM, axC, axD)
            vlines!(ax, e; linestyle = :dot, alpha = 0.35, color = :gray30)
        end
    end
end
f
save(joinpath(@__DIR__, "slop_cut_variants.pdf"), f)

# A single summary figure of the headline numbers.
fs = Figure(size = (900, 400))
names = [string(r.var.name) for r in solutions]
axp = CairoMakie.Axis(fs[1, 1]; title = "profit", xticks = (1:length(names), names))
barplot!(axp, 1:length(names), [r.profit for r in solutions]; color = line_colors[1:length(names)])
axt = CairoMakie.Axis(fs[1, 2]; title = "total batch time", xticks = (1:length(names), names))
barplot!(axt, 1:length(names), [r.T for r in solutions]; color = line_colors[1:length(names)])
fs
save(joinpath(@__DIR__, "slop_cut_variants_summary.pdf"), fs)

# ## Closing remarks
#
# * The recycle DAE is the exact `slopint.cpp` model: 29 states, `NU = 15` controls
#   (`R`, 12 valve flows, 2 pumps) collapsed here to `R` plus twelve per-tray flows, and the
#   extended algebraic system 7.28. Its correctness is established in
#   `context/slop-cut/recycle_proto.jl`, where the reference controls reproduce every
#   differential state of `DAT/slopint.dat` and the total time exactly.
# * The periodic closure that makes the recycle *self-consistent* — the holdup and
#   composition arriving in the reservoir equal the slop cut produced by the current batch —
#   is what the MUSCOD-II `rcfcn_*` constraints enforce. Here it is relaxed: the recycle
#   composition is fixed to the published optimum and the recycled amount is the free
#   reservoir initial holdup `m_\ell(0)`. This keeps the example a single batch, at the cost
#   of an approximate comparison with the published scenario-B/C optima.
# * The discrete valve choice (one tray open at a time, an SOS1 condition) is relaxed to
#   continuous per-tray flows in variant C. The published optimum is essentially regenerated
#   if the routing is fixed to the reference (tray 3 for ``S_1`` and ``S_2``) — the
#   "valves-fixed" scenario of Sager.
#
# ## References
#
# * Sager, S. *Numerical methods for mixed-integer optimal control problems*. PhD thesis,
#   University of Heidelberg, 2005 (§7.3, slop-cut benchmark with recycling).
# * The MUSCOD-II benchmark sources `slopint.cpp`, `slopint.dat`, `slop.dat`.
# * `examples/slop_cut/main.jl` — the base problem this file extends.
