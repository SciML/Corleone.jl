#src ---
#src title: Switching-Time Optimization of the Subway Ride with Point and Path Constraints
#src description: Energy-optimal subway operation with a speed restriction at a track point and on a track section, solved as a switching-time optimization
#src tags:
#src   - Multi-stage
#src   - Optimal Control
#src   - Switching time
#src   - Subway
#src icon: 🚇
#src ---

# ## Switching-Time Optimization of the Subway Ride with Point and Path Constraints

# This example extends [the subway switching-time problem](@ref) by **state constraints**:
#
# 1. a **point constraint**: the train must not exceed a speed limit ``v_4`` at the track
#    point ``S_4``,
# ```math
# x_0(\tilde t_4) = S_4, \qquad x_1(\tilde t_4) \le v_4,
# ```
# 2. a **path constraint**: the train must not exceed a speed limit ``v_5`` on the track
#    section ``x_0 \ge S_5`` (variant 3 below),
# ```math
# x_1(t) \le v_5 \quad \text{if} \quad x_0(t) \ge S_5.
# ```
#
# Such restrictions model, for example, sections of track that are not in the best shape.
# Because ``x_0`` is monotone, both can be handled **exactly** (no constraint discretization)
# by **splitting the shooting chain**: a stage boundary is placed where the train passes the
# critical track point, and the constraint conditions become terminal constraints of the
# corresponding stages. The rest of the machinery — normalized stages, duration parameters
# scaling the right-hand side, and a ``\sum_i T_i = t_f`` coupling — is identical to the
# base problem.

# ## Setup

using Corleone
using Lux
using LuxCore
using Random
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using CairoMakie

# ## Model

# ### Parameters

const GAMMA = 3600.0 / 5280.0      # unit conversion between mph and ft/s
const W = 78000.0                  # weight [lbs]
const WEFF = W + 72000.0 * 0.1     # effective weight [lbs]
const AREA = 100.0                 # frontal area [ft^2]
const BCOEF = 0.045
const CDRAG = 0.000546
const COAST = 0.367                # constant coasting deceleration term
const GC = 32.2                    # gravitational acceleration [ft/s^2]
const EFF = 1.0                    # efficiency

const V1 = 0.979474 / GAMMA        # [ft/s]
const V2 = 6.73211 / GAMMA
const V3 = 14.2658 / GAMMA

const A1 = 6017.611205
const A2 = 12348.34865
const A3 = 11124.63729
const P1 = 106.1951102
const P2 = 180.9758408
const P3 = 354.136479

const B_SERIES = [
    -0.1983670410e2, 0.1952738055e3, 0.2061789974e4,
    -0.7684409308e3, 0.2677869201e3, -0.3159629687e2,
]
const B_PARALLEL = [
    -0.1577169936e3, 0.3389010339e4, 0.6202054610e4,
    -0.4608734450e4, 0.2207757061e4, -0.3673344160e3,
]
const C_SERIES = [
    0.3629738340e2, -0.2115281047e3, 0.7488955419e3,
    -0.9511076467e3, 0.5710015123e3, -0.1221306465e3,
]
const C_PARALLEL = [
    0.4120568887e2, 0.3408049202e3, -0.1436283271e3,
    0.8108316584e2, -0.5689703073e1, -0.2191905731e1,
]

poly(cs, z) = cs[1] + z * (cs[2] + z * (cs[3] + z * (cs[4] + z * (cs[5] + z * cs[6]))))

running_resistance(v) = let vm = GAMMA * v
    CDRAG * AREA * vm^2 + BCOEF * W * vm / 2000 + 1.3 / 2000 * W + 116
end

tractive_series(v) = let z = 1.0 / (0.1 * GAMMA * v - 0.3)
    poly(B_SERIES, z)
end
tractive_parallel(v) = let z = 1.0 / (0.1 * GAMMA * v - 1.0)
    poly(B_PARALLEL, z)
end

# ### Decoupled modes

function decoupled_rhs(mode, v)
    if mode == 1
        if v <= V1
            return GC * EFF * A1 / WEFF, P1
        elseif v <= V2
            return GC * EFF * A2 / WEFF, P2
        else
            dv = GC * (EFF * tractive_series(v) - running_resistance(v)) / WEFF
            dE = poly(C_SERIES, 1.0 / (0.1 * GAMMA * v))
            return dv, dE
        end
    elseif mode == 2
        if v <= V2
            return GC * EFF * A3 / WEFF, P3
        elseif v <= V3
            return GC * EFF * A3 / WEFF, P3
        else
            dv = GC * (EFF * tractive_parallel(v) - running_resistance(v)) / WEFF
            dE = poly(C_PARALLEL, 1.0 / (0.1 * GAMMA * v - 1.0))
            return dv, dE
        end
    elseif mode == 3
        return -(GC * GAMMA) * running_resistance(v) / WEFF - COAST, 0.0
    else
        return -4.4, 0.0   # maximal braking
    end
end

struct SubwayMode{M} end

function (::SubwayMode{M})(du, u, p, t) where {M}
    T = p[1]
    v = u[2]
    dv, dE = decoupled_rhs(M, v)
    du[1] = T * v
    du[2] = T * dv
    du[3] = T * dE / 3600.0
    return
end

const TF = 65.0
const S = 2112.0
const U0 = [0.0, 0.0, 0.0]

carry_state = StageTransition((traj, _) -> copy(last(traj.u)))

# ## Solving a constrained variant

# `solve_variant` builds the shooting chain for a fixed mode sequence. The stage boundary
# with index `split` is placed where the train passes the critical track point `S4`, and
# the constrained conditions become terminal constraints of that stage:
# ``x_0 = S_4`` and ``x_1 \le v_4``. Additional per-stage constraints (used by the path
# constraint variant below) can be passed via `extra_cons` in the same
# `(:symbol, stage) => (t, bounds)` format.
#
# Neither the split index nor the active mode sequence is hand-supplied. Following the
# reference (Sager 2005, section 7.1), we only assume that the critical track point is
# passed "within the stage ``[\tilde t_3, \tilde t_5]``" — the phase between the first
# parallel arc and the final coasting/braking phase — and that the mode sequence is a
# *superset* of the optimal one. `solve_variant_window` then enumerates all candidate
# splits inside that window (optionally several initial guesses per split) and keeps the
# cheapest feasible solution; unneeded modes are switched off by the optimizer itself,
# driving their durations to zero.

function solve_variant(modes, split, S4, v4, d0 = nothing, extra_cons = ())
    N = length(modes)
    d0 = d0 === nothing ? fill(TF / N, N) : d0
    stages = [
        SingleShootingLayer(
            ODEProblem(
                SubwayMode{modes[i]}(), U0, (0.0, 1.0), [d0[i]];
                abstol = 1.0e-12, reltol = 1.0e-10
            ),
            Tsit5(); bounds_p = ([0.0], [TF]),
            parameter_initialization = ((rng, problem, tunable_p, bounds) -> [d0[i]])
        ) for i in 1:N
    ]
    layers = Any[]
    for i in 1:N
        push!(layers, stages[i])
        i < N && push!(layers, carry_state)
    end
    chain = Chain(layers...)

    stage_names = [Symbol("layer_$(2i - 1)") for i in 1:N]
    duration_constraint = (ps -> [sum(getproperty(ps, nm).p[1] for nm in stage_names)]) =>
        (lb = [TF], ub = [TF])

    constraints = (
        (:x₁, N) => (t = [1.0], bounds = (S, S)),
        (:x₂, N) => (t = [1.0], bounds = (0.0, 0.0)),
        (:x₁, split) => (t = [1.0], bounds = (S4, S4)),
        (:x₂, split) => (t = [1.0], bounds = (-Inf, v4)),
        extra_cons...
    )

    optprob = OptimizationProblem(
        chain, AutoForwardDiff(), Val(:ComponentArrays);
        loss = :x₃, constraints = constraints,
        parameter_constraints = duration_constraint
    )
    sol = solve(
        optprob, Ipopt.Optimizer();
        hessian_approximation = "limited-memory",
        max_iter = 5000, tol = 1.0e-8, print_level = 0
    )
    success = sol.retcode == SciMLBase.ReturnCode.Success
    durations = nothing
    if success
        ps, st = LuxCore.setup(Random.default_rng(), chain)
        ps_opt = sol.u + zero(ComponentArray(ps))
        durations = [getproperty(ps_opt, nm).p[1] for nm in stage_names]
        # Ipopt with limited-memory Hessian may converge with durations a few multiples of
        # the machine epsilon below their lower bound (e.g. -5e-9); that is zero for all
        # practical purposes, so we clamp before checking feasibility.
        durations = max.(durations, 0.0)
        success = sum(durations) <= TF + 1.0e-6
    end
    return sol, durations, chain, success
end

"""
    solve_variant_window(modes, window, S4, v4; inits = nothing, extra_cons = nothing)

Enumerate all splits of the point constraint into the stages of `window` (the reference's
``[\\tilde t_3, \\tilde t_5]``), optionally with several initial duration guesses per split
(`inits`) and additional per-stage constraints. `extra_cons` may be a tuple of constraints
or a function `split => constraints` evaluated for each candidate split.

Returns the best feasible solution, its durations, its chain, the discovered split index,
and the list of all candidates for inspection.
"""
function solve_variant_window(modes, window, S4, v4; inits = nothing, extra_cons = nothing)
    N = length(modes)
    uniform = fill(TF / N, N)
    inits = inits === nothing ? [uniform] : inits
    best = nothing
    best_durs = nothing
    best_chain = nothing
    best_split = 0
    candidates = NamedTuple[]
    for split in window, d0 in inits
        ex = extra_cons isa Function ? extra_cons(split) : something(extra_cons, ())
        sol, durations, chain, success = solve_variant(modes, split, S4, v4, d0, ex)
        feasible = success && durations !== nothing &&
            all(d -> d >= 0.0, durations) && sum(durations) <= TF + 1.0e-6
        push!(candidates, (; split, energy = feasible ? sol.objective : Inf))
        if feasible && (best === nothing || sol.objective < best.objective)
            best, best_durs, best_chain, best_split = sol, durations, chain, split
        end
    end
    return best, best_durs, best_chain, best_split, candidates
end

# ## Dense simulation

const INACTIVE = 1.0e-9

function simulate(durations, modes; saveat = 0.01)
    t_all = Float64[]
    x_all = Vector{Float64}[]
    w_all = Float64[]
    switches = Float64[]
    t0 = 0.0
    state = copy(U0)
    for i in eachindex(modes)
        Ti = max(durations[i], 0.0)
        Ti < INACTIVE && continue
        prob = ODEProblem(SubwayMode{modes[i]}(), state, (t0, t0 + Ti), [1.0])
        sol = solve(prob, Tsit5(); saveat = saveat, abstol = 1.0e-11, reltol = 1.0e-11)
        for (t, u) in zip(sol.t, sol.u)
            (isempty(t_all) || t > t_all[end]) || continue
            push!(t_all, t)
            push!(x_all, u)
            push!(w_all, Float64(modes[i]))
        end
        tend = t0 + Ti
        if t_all[end] < tend - 1.0e-12
            push!(t_all, tend)
            push!(x_all, sol.u[end])
            push!(w_all, Float64(modes[i]))
        end
        state = sol.u[end]
        t0 = tend
        push!(switches, t0)
    end
    return t_all, x_all, w_all, switches
end

function plot_variant(t, position, velocity_mph, energy, mode_traj, switches, S4, v4)
    inner = switches[1:(end - 1)]
    f = Figure(size = (1000, 750))

    ax1 = CairoMakie.Axis(f[1, 1], title = "Position", xlabel = "t [s]", ylabel = "x₀ [ft]")
    lines!(ax1, t, position; linewidth = 2)
    hlines!(ax1, S4; color = :red, linestyle = :dash)
    for s in inner
        vlines!(ax1, s; color = :black, linestyle = :dot, alpha = 0.4)
    end

    ax2 = CairoMakie.Axis(f[1, 2], title = "Velocity", xlabel = "t [s]", ylabel = "x₁ [mph]")
    lines!(ax2, t, velocity_mph; linewidth = 2, color = :orange)
    hlines!(ax2, v4 * GAMMA; color = :red, linestyle = :dash)
    for s in inner
        vlines!(ax2, s; color = :black, linestyle = :dot, alpha = 0.4)
    end

    ax3 = CairoMakie.Axis(
        f[2, 1], title = "Mode", xlabel = "t [s]", ylabel = "w";
        yticks = 1:4, limits = (nothing, (0.5, 4.5))
    )
    stairs!(ax3, t, mode_traj; linewidth = 2, step = :post)
    for s in inner
        vlines!(ax3, s; color = :black, linestyle = :dot, alpha = 0.4)
    end

    ax4 = CairoMakie.Axis(f[2, 2], title = "Energy", xlabel = "t [s]", ylabel = "energy")
    lines!(ax4, t, energy; linewidth = 2, color = :green)
    return f
end

# ## Variant 1: point constraint at ``S_4 = 1200``

const V4 = 22.0 / GAMMA   # [ft/s]

# The superset sequence below contains the reference structure
# ``1,2,1,3,4,2,1,3,4`` (Sager 2005, eq. (7.6)) with one spare braking stage. The optimizer
# reproduces the reference structure unaided: the discovered split is the stage in which
# the train passes ``S_4``, and all durations come out active.

modes1 = [1, 2, 1, 3, 4, 2, 1, 3, 4]
window1 = 4:6                       # [t̃₃, t̃₅]: the first parallel arc .. coast
sol1, durs1, _, split1, cands1 = solve_variant_window(modes1, window1, 1200.0, V4)
t1, x1, w1, sw1 = simulate(durs1, modes1)

println("== S4 = 1200, v4 = 22 ==")
println("Discovered split:  stage $split1 of window $window1")
println("Candidate energies: ", [c.energy for c in cands1])
println("Energy:            ", sol1.objective, "   (reference 1.3978)")
println("Status:            ", sol1.retcode)
println("Active switch t:   ", round.(sw1[1:(end - 1)]; digits = 4))

fig1 = plot_variant(t1, [u[1] for u in x1], [u[2] * GAMMA for u in x1], [u[3] for u in x1], w1, sw1, 1200.0, V4)
save("subway_variant1_S4_1200.pdf", fig1)

# ## Variant 2: point constraint at ``S_4 = 700``

# The same window now also has to *discover the structure*: for the earlier constraint the
# braking stage of the superset is switched off (zero duration), collapsing the sequence to
# the reference structure ``1,2,1,3,2,1,3,4`` (Sager 2005, eq. (7.7)).

modes2 = [1, 2, 1, 3, 2, 1, 3, 4]
window2 = 3:5                       # [t̃₃, t̃₅]: first parallel arc .. coast
sol2, durs2, _, split2, cands2 = solve_variant_window(modes2, window2, 700.0, V4)
t2, x2, w2, sw2 = simulate(durs2, modes2)

println("== S4 = 700, v4 = 22 ==")
println("Discovered split:  stage $split2 of window $window2")
println("Candidate energies: ", [c.energy for c in cands2])
println("Energy:            ", sol2.objective, "   (reference 1.32518)")
println("Status:            ", sol2.retcode)
println("Active switch t:   ", round.(sw2[1:(end - 1)]; digits = 4))

fig2 = plot_variant(t2, [u[1] for u in x2], [u[2] * GAMMA for u in x2], [u[3] for u in x2], w2, sw2, 700.0, V4)
save("subway_variant2_S4_700.pdf", fig2)

# ## Variant 3: path constraint ``x_1 \le v_5`` on ``x_0 \ge S_5``

const V5 = 24.0 / GAMMA   # [ft/s] speed limit, ~ 35.2 ft/s
const S5 = 1200.0         # [ft] start of the restricted track section

# A path constraint seems to require a densely sampled trajectory — but on this problem it
# does not, for two structural reasons:
#
# 1. **One interior point instead of a region.** ``x_0`` is strictly monotonically
#    increasing, so ``x_0 \ge S_5`` has a single entry time: the condition
#    ``x_0(t) = S_5`` is an interior point (equality) constraint, exactly like the split
#    condition of the variants above.
# 2. **Monotone velocity within each stage.** Series/parallel accelerate, coast/brake
#    decelerate, so ``x_1 \le v_5`` holds on a whole stage iff it holds at the end of the
#    stage. A one-sided terminal bound ``x_1 \le v_5`` on the *accelerating* stages after
#    the entry therefore enforces the path constraint **exactly** — no interior sampling
#    points, no grid convergence study.
#
# The optimal structure changes qualitatively, though: the constrained section is entered
# too fast to coast through it, so an **interior braking arc** is required. The integer
# solution of the reference (Sager 2005, eq. (7.11)) switches to braking strictly below the
# limit and then re-accelerates so that the velocity *touches* ``v_5`` exactly once before
# the final coast and brake:
#
# ```math
# w(t) = S(1, 2, 1, 3, 4, 1, 3, 4;\; 2.68054,\; 13.8253,\; 12.2412,\; 4.03345,\;
# \; 1.65001,\; 15.3543,\; 7.99192,\; 7.22329), \qquad \Phi_1 = 1.38367.
# ```
#
# We supply this structure as a superset (one extra coasting stage) and discover both the
# split stage and the zeroed stage via enumeration:

modes3 = [1, 2, 1, 3, 4, 1, 3, 4, 3]   # superset of (7.11) with one spare coasting stage
window3 = 3:5                          # [t̃₃, t̃₅] of the (7.11) structure

# Terminal velocity bound ``x_1 \le v_5`` on every accelerating stage after the entry:

extra3 = split -> [
    (:x₂, j) => (t = [1.0], bounds = (-Inf, V5))
    for j in (split + 1):8 if modes3[j] == 1 || modes3[j] == 2
]

# Initial guesses: uniform plus the reference durations of (7.6) and (7.11), embedded into
# 9 stages. Enumeration is over splits × initial guesses.

inits3 = [
    fill(TF / 9, 9),
    [2.86362, 10.722, 15.3108, 5.81821, 1.18383, 2.72451, 12.917, 5.47402, 7.98594],
    [2.68054, 13.8253, 12.2412, 4.03345, 1.65001, 15.3543, 7.99192, 7.22329, 0.0],
]

sol3, durs3, _, split3, cands3 = solve_variant_window(
    modes3, window3, S5, V5; inits = inits3, extra_cons = extra3
)
t3, x3, w3, sw3 = durs3 === nothing ? ([], [], [], []) : simulate(durs3, modes3)

# dense verification of the path constraint on the restricted section:
vmax3 = maximum(u[2] for (u, t) in zip(x3, t3) if u[1] >= S5)

println("== S5 = 1200, v5 = 24 mph (path constraint) ==")
println("Discovered split:  stage $split3 of window $window3")
println("Candidate energies: ", [c.energy for c in cands3])
println("Energy:            ", sol3.objective, "   (reference 1.38367)")
println("Status:            ", sol3.retcode)
println("max v on section:  ", vmax3 * GAMMA, " mph   (limit 24)")
println("Active switch t:   ", round.(sw3[1:(end - 1)]; digits = 4))

fig3 = plot_variant(t3, [u[1] for u in x3], [u[2] * GAMMA for u in x3], [u[3] for u in x3], w3, sw3, S5, V5)
save("subway_variant3_path_constraint.pdf", fig3)

# ## Discussion

# All three variants reproduce the reference results — energies ``1.3978`` (``S_4 = 1200``),
# ``1.32518`` (``S_4 = 700``) and ``1.38367`` for the path constraint — without
# hand-supplied switching structures:
#
# - the **mode sequence** is only a superset; the optimizer switches unneeded modes off by
#   driving their durations to zero (structure discovery, as in the base example),
# - the **stage containing the critical track point** is discovered by enumerating the
#   candidates inside the reference's window ``[\tilde t_3, \tilde t_5]``,
# - the **path constraint** is enforced *exactly* through the split construction and
#   terminal stage bounds — exploiting monotonicity of ``x_0`` and of the velocity within
#   each stage — instead of being discretized on a grid.
#
# What a switching-time method cannot do is discover the mode *order* from an unstructured
# start; that requires the convexified relaxation approach of Sager (2005). The window
# ``[\tilde t_3, \tilde t_5]`` and the superset sequences remain structural assumptions —
# deliberately weak ones, identical in spirit to those of the reference.
