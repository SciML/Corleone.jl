#src ---
#src title: Optimization of the Slop-Cut Batch Distillation Problem
#src description: Multiple-shooting optimization of a multi-stage DAE model of a ternary batch distillation with production and slop cuts
#src tags:
#src   - Multi-stage
#src   - DAE
#src   - Multiple shooting
#src   - Batch distillation
#src   - Switching time
#src icon: ⚗️
#src ---

# ## Optimization of the Slop-Cut Batch Distillation Problem

# This tutorial **optimizes** the *slop-cut* batch distillation problem of Sager,
# *Numerical methods for mixed-integer optimal control problems*, PhD thesis (2005),
# which goes back to the MUSCOD-II batch distillation benchmark.
#
# A ternary mixture is distilled in a column with five trays, a reboiler (the *pot*) and a
# total condenser. The batch is split into two *production cuts* ``P_1, P_2`` separated by
# two *slop cuts* ``S_1, S_2``. The manipulated variable is the **reflux ratio** ``R``,
# which here is a piecewise-constant *staircase* on a 20-node grid inside every cut; each cut
# also has a free duration ``T_k``.
#
# The objective is the profit
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
using ForwardDiff
using SparseArrays
using CairoMakie

# ### Model constants

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

# The start values in `slop.dat` are inconsistent, so the algebraic state is obtained once
# from a Newton solve.

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
# accumulates ``\int T\,d\tau``. A type parameter `F` marks the slop cut `S2`, in which the
# pot fraction `x1d` is frozen.

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

# ### Optimization problem setup

# Optimize over a 20-node reflux staircase by default; `NODES` and `MAXITER` can be
# overridden from the environment (`SLOP_NODES`, `SLOP_MAXITER`) to experiment with the
# discretization and the number of Ipopt iterations.
const NODES = parse(Int, get(ENV, "SLOP_NODES", "5"))
const MAXITER = parse(Int, get(ENV, "SLOP_MAXITER", "5"))

# The staircase breakpoints. `CONTROL_GRID` are the nodes at which the reflux ratio is a
# free variable. Because a `SingleShootingLayer` splits its time span at every control
# breakpoint, the shooting layer saves the trajectory *exactly* at these nodes (plus the
# stage endpoints). `CONS_T` is therefore derived from the same grid, so the path
# constraints follow whatever discretization is used. This matters because
# `CorleoneDynamicOptProblem` matches a constraint time against the saved times with
# `findall(∈(specs.t), tk)` and *silently drops* any constraint whose time does not
# coincide with a saved point.
const CONTROL_GRID = collect(range(0.0, 1.0; length = NODES + 1))[1:NODES]
const CONS_T = sort(unique(vcat(CONTROL_GRID, 1.0)))

function reflux_ctrl(R0)
    return ControlParameter(CONTROL_GRID; name = :R, bounds = (0.3, 15.0), controls = collect(R0))
end

# One cut as a `SingleShootingLayer`. `F=true` freezes `x1d` (the slop cut `S2`).
# The integration is pinned to a fixed step (`adaptive=false, dt=5e-4`), which is the
# forward setting that makes the sensitivities of this stiff DAE well behaved.

const SOLVE_KWARGS = (; adaptive = false, dt = 5.0e-4)

function mkstage(F, R0, T0; Tlb = 0.10, Tub = 1.70)
    u, du = initstate(R0[1])
    prob = DAEProblem(
        SlopDAE{F}(), du, u, (0.0, 1.0), [R0[1], T0, V0, PRES];
        differential_vars = DV
    )
    return SingleShootingLayer(
        prob, DFBDF(); controls = (1 => reflux_ctrl(R0),), params = [2],
        bounds_p = ([Tlb], [Tub]), dae_initialization = BrownFullBasicInit(),
        SOLVE_KWARGS...
    )
end

# ### Coupling the cuts — multiple shooting

# Single shooting over the four coupled DAE stages makes the end-to-end sensitivities
# ill-conditioned and Ipopt stalls. Instead the cuts are coupled with `StageLinkage`: each
# linkage owns the next cut's interface state `u0` (a free, bounded variable) and the
# coupling `u0 - x_f(prev) = 0` is imposed as an equality constraint. The second linkage
# additionally resets the distillate accumulators before the production cut `P2`.

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

function build_chain(T0, R; u0s = nothing)
    s1 = mkstage(false, R[1], T0[1])
    s2 = mkstage(false, R[2], T0[2])
    s3 = mkstage(false, R[3], T0[3])
    s4 = mkstage(true, R[4], T0[4])
    l1 = link_carry()
    l2 = link_reset()
    l3 = link_carry()
    if !isnothing(u0s)
        l1.u0 .= u0s[1]; l2.u0 .= u0s[2]; l3.u0 .= u0s[3]
    end
    return Chain(s1, l1, s2, l2, s3, l3, s4)
end

# The constraints and the profit loss. The purities are imposed in their linearised form
# (`x25 - 0.98 x24 ≥ 0`, `x26 - 0.96 x24 ≥ 0`, `x2 + x3 ≤ 0.01`), and the pot holdup is kept
# above a small floor.

const P1QL = 0.98
const P2QL = 0.96
const P3QL = 0.99
const CPRICE = 4.5
const FEEDNM = 100.0
const M_MIN = 2.0
const CP = CPRICE / FEEDNM

function make_problem(chain; kwargs...)
    constraints = (
        (:(x₂₅ - 0.98 * x₂₄), 1) => (t = [1.0], bounds = (0.0, Inf)),
        (:(x₂₆ - 0.96 * x₂₄), 3) => (t = [1.0], bounds = (0.0, Inf)),
        (:(x₂ + x₃), 4) => (t = [1.0], bounds = (-Inf, 1.0 - P3QL)),
        (:x₁, 1) => (t = CONS_T, bounds = (M_MIN, Inf)),
        (:x₁, 2) => (t = CONS_T, bounds = (M_MIN, Inf)),
        (:x₁, 3) => (t = CONS_T, bounds = (M_MIN, Inf)),
        (:x₁, 4) => (t = CONS_T, bounds = (M_MIN, Inf)),
    )
    loss = [(:(-$(CP) * x₂₄), 1), (:(-$(CP) * x₂₄), 3), (:(x₂₇ - $(CP) * x₁), 4)]
    return OptimizationProblem(
        chain, AutoForwardDiff(), Val(:ComponentArrays);
        loss = loss, constraints = constraints, kwargs...
    )
end

# ### Warm start

# The warm start is the best solution found so far: profit ``0.4444`` at ``T = 3.4978``
# with 20 reflux nodes per cut (`context/slop-cut/conv_fixed_20.txt`). It was obtained with
# the fixed-step forward setting above and Ipopt's limited-memory (L-BFGS) Hessian
# approximation.

const WARM_T = [1.1746618894595116, 0.5689127497722057, 1.1533048568406312, 0.6009589878645355]

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

const WARM_U0 = [
    [70.5463209732873, 0.05855977538147307, 0.4594625869434197, 0.48197763767510726,
        251.88143365748365, 0.11394840394631674, 0.646379427565887, 239.1018823862979,
        0.22875148489950775, 0.6868243950416419, 227.4610110205777, 0.4235059164753662,
        0.5551052118191685, 216.13206070273336, 0.6593673221249581, 0.33676352436096246,
        206.34583508554033, 0.8445856564861929, 0.1549024152109246, 200.1711555776741,
        0.9437592530748535, 0.05619050461286092, 5.024231228556938e-5, 29.453679026710684,
        28.864660707812796, 0.5885968003272682, 1.1746618894594822],
    [64.41497828604018, 0.006031871670631358, 0.4661751246792889, 0.5277930036500798,
        259.30355831179537, 0.008271664771141255, 0.6884433464656288, 250.21728021311645,
        0.012148435018740095, 0.8503476102257231, 244.22978550034208, 0.021129451873535453,
        0.9258349372355507, 240.88866856462505, 0.045234026765068706, 0.9361615305045903,
        238.04206787249655, 0.10908704819638156, 0.8849524319418, 233.18379128047718,
        0.2582100616990739, 0.7402006755186183, 0.0015892627823076895, 0.0, 0.0, 0.0,
        1.7435746392318914],
    [37.20850662356405, 2.3464675571225297e-7, 0.10496092203082776, 0.8950388433224165,
        278.40545101374283, 2.552423218668538e-7, 0.15055739205018223, 275.8354506925021,
        3.135400444754552e-7, 0.242867783581212, 270.8881095268058, 5.038628367133212e-7,
        0.4033028853323639, 263.06904765034574, 1.1787583638388285e-6, 0.6157572631867113,
        254.08261858823585, 3.508475772129685e-6, 0.8075770312359192, 247.10220305253713,
        1.1047519453815234e-5, 0.9247146226738683, 0.07527432980667796, 27.206471662476215,
        0.3884195715710217, 26.11827108647094, 2.8968794960727173],
]

# ### Solve

function resample_staircase(v, t_new)
    n = length(v)
    t_old = collect(range(0.0, 1.0; length = n + 1))[1:n]
    return [v[clamp(searchsortedlast(t_old, t), 1, n)] for t in t_new]
end

# Resample the 20-node warm start onto the current control grid so the tutorial can be run
# with any `NODES`. On the original grid this is the identity.
warm_R = [resample_staircase(WARM_R[k], CONTROL_GRID) for k in 1:4]

chain = build_chain(WARM_T, warm_R; u0s = WARM_U0)
ps, st = LuxCore.setup(Random.default_rng(), chain)
x0 = ComponentArray(ps)

# The constraint Jacobian is sparse. Provide a `cons_jac_prototype` so that OptimizationMOI
# reports only its structurally nonzero entries to Ipopt. The prototype is obtained once
# from an `ForwardDiff` Jacobian of the constraint function at the warm start; the off-diagonal
# structure is what matters, the values are irrelevant.
op0 = make_problem(chain)
cons0 = op0.f.cons
mcons = length(op0.lcons)
Jd = ForwardDiff.jacobian(x -> begin
    r = zeros(eltype(x), mcons)
    cons0(r, x, st)
    r
end, x0)
cons_jac_prototype = sparse(Jd .!= 0)

op = make_problem(chain; cons_jac_prototype = cons_jac_prototype)
println("variables = ", length(x0), ", constraints = ", length(op.lcons),
    ", Jacobian nnz = ", nnz(cons_jac_prototype))

# Ipopt with the **limited-memory** (L-BFGS) Hessian approximation. For experimentation we
# only run a handful of iterations from the warm start.

sol = solve(
    remake(op, u0=sol.u), Ipopt.Optimizer();
    #hessian_approximation = "limited-memory",
    max_iter = 25, tol = 1.0e-6, constr_viol_tol = 1.0e-6,
    mu_strategy = "adaptive", bound_relax_factor = 1.0e-8, print_level = 5
)
println("retcode = ", sol.retcode, ", objective = ", sol.objective)

# ### Evaluate the optimized solution

xopt = sol + zero(x0)
trajs = Corleone._forward_collect_stage_trajs(chain, xopt, st)
e1, e2, e3, e4 = (last(trajs[i].u) for i in 1:4)
P1 = e1[24]
P2 = e3[24]
P3 = e4[1]
T = e4[27]
profit = CPRICE * (P1 + P2 + P3) / FEEDNM - T

println("profit = ", round(profit; digits = 5), "   T = ", round(T; digits = 5))
println("P1 = ", round(P1; digits = 4), "  P2 = ", round(P2; digits = 4), "  P3 = ", round(P3; digits = 4))
println("pur1 = ", round(e1[25] / e1[24]; digits = 6),
    "  pur2 = ", round(e3[26] / e3[24]; digits = 6),
    "  x3 = ", round(1 - e4[2] - e4[3]; digits = 6))

opt_T = [getproperty(xopt, Symbol("layer_$(2i - 1)")).p[1] for i in 1:4]
opt_R = [collect(getproperty(xopt, Symbol("layer_$(2i - 1)")).controls) for i in 1:4]
println("durations = ", round.(opt_T; digits = 4))

# ### Plot the optimized reflux staircase

f = Figure(size = (900, 500))
ax = CairoMakie.Axis(f[1, 1]; title = "Optimized reflux staircase", xlabel = "cut", ylabel = "R")
for k in 1:4
    stairs!(ax, collect(0:NODES) .+ 4(k - 1), vcat(opt_R[k], opt_R[k][end]);
        step = :post, linewidth = 2, label = ["P1", "S1", "P2", "S2"][k])
end
axislegend(ax; position = :lt)
f
