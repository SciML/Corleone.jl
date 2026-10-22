#src ---
#src title: Optimal Control of a Two Stage Reactor System
#src description: Two optimal-control stages joined by a parameterized mixing transition, free stage durations and a terminal state constraint
#src tags:
#src   - Multi-stage
#src   - Optimal Control
#src   - Reactor
#src icon: ⚗️
#src ---

# ## Optimal Control of a Two-Stage Reactor System

# This tutorial reproduces the two-stage **batch reactor** optimal-control problem of
# Hirmajer & Fikar, *Optimal Control of a Two-Stage Reactor System*, Chem. Pap.
# 60(5):381--387 (2006), DOI: [10.2478/s11696-006-0069-x](https://doi.org/10.2478/s11696-006-0069-x).
#
# A connected system of two batch reactors is operated in two stages. In the first,
# **non-isothermal** stage the chain reaction ``A \to B \to C`` proceeds and the reactor
# temperature is the control. When this stage is finished, a volume ``S`` of a dilute
# aqueous solution of ``B`` (concentration ``c_B^s``) is added. The mixture (volume
# ``V_2 = V_1 + S``) is then loaded into a second reactor where three **parallel**
# reactions ``B \to D``, ``B \to E`` and ``2B \to F`` take place at a fixed temperature.
#
# We want to
#
# ```math
# \max_{S, \Delta t_i, T[0, t_s]} \quad J_0 = V_2\, c_D(t_P)
# ```
#
# subject to a minimum final concentration of ``D`` and a maximum processing time
#
# ```math
# J_1 = c_D(t_P) - c_D^w \ge 0, \qquad
# J_2 = t_P - \sum_i \Delta t_i \ge 0 .
# ```
#
# The two stages live on normalized time ``\tau \in [0, 1]`` and their right-hand sides are scaled
# by the (optimized) stage durations ``T_1`` and ``T_2``. The added volume enters as the
# parameter of the `StageTransition`, which applies the **mixing** relations of the paper.

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
using SciMLSensitivity

# ## Model

# The six species ``A,\dots,F`` are tracked as concentrations ``[\mathrm{mol\,m^{-3}}]``.
# Two auxiliary states are carried along to make the objective a plain terminal state:
#
# - ``x_7 = V\,c_D``, the **total amount of D** (in mol). Because the volume is constant
#   within each stage, ``\mathrm{d}(V c_D)/\mathrm{d}t = V\,\mathrm{d}c_D/\mathrm{d}t``.
#   Since ``c_D = 0`` in stage 1, only the second stage produces it.
# - ``x_8 = V_2``, the **stage-2 volume**, which is needed in the right-hand side of the
#   second stage and is set (= ``V_1 + S``) at the switch.
#
# ### Stage 1: non-isothermal, ``A \to B \to C``
#
# The Arrhenius rate constants are
# ``k_1(T) = 0.0444\,e^{-2500/T}`` and ``k_2(T) = 6889\,e^{-5000/T}``, with ``T`` the
# reactor temperature (the control). The interval duration ``T_1`` scales the right-hand side.

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

# ### Stage 2: isothermal, ``B \to D``, ``B \to E``, ``2B \to F``
#
# The volume ``V_2`` is read from state 8, and the amount of ``D`` grows as
# ``\mathrm{d}(V_2 c_D)/\mathrm{d}t = V_2 \cdot 0.02\,c_B``.

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

# ### Switch: the mixing operation
#
# At the switching time ``t_s`` the mixing equations of the paper apply:
#
# ```math
# V_2 c_A(t^+) = V_1 c_A(t^-), \qquad
# V_2 c_B(t^+) = V_1 c_B(t^-) + S\,c_B^s, \qquad
# V_2 c_C(t^+) = V_1 c_C(t^-),
# ```
#
# with ``V_2 = V_1 + S`` and the same (dilution) relation for ``c_D, c_E, c_F``. The amount
# of ``D`` (``x_7``) is unchanged and ``x_8`` is set to ``V_2``.

const V1 = 0.1
const c_b_s = 600.0

function switch_at_ts(traj, p)
    A, B, C, D, E, F, D_amount, V2 = last(traj.u)
    S = p[1]
    V2_new = V1 + S
    return [
        V1 * A / V2_new, (V1 * B + S * c_b_s) / V2_new, V1 * C / V2_new,
        V1 * D / V2_new, V1 * E / V2_new, V1 * F / V2_new,
        V1 * D,    # amount of D is preserved across the switch
        V2_new,    # carry the stage-2 volume
    ]
end

# ## Defining the Stages

# The initial concentration of ``A`` in the first reactor is ``c_A(0) = 2000\,\mathrm{mol\,m^{-3}}``.
# The initial durations emulate the paper's guess of five 15-minute control intervals for
# the first stage and a 15-minute second stage, so ``T_1 = 75`` and ``T_2 = 15`` min.

u0 = [2000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

const t_p = 180.0   # processing time [min]
const c_d_w = 150.0 # minimum final concentration of D [mol/m^3]

T1_init = 5.0 * 15.0
T2_init = 15.0

prob1 = ODEProblem(stage1!, u0, (0.0, 1.0), [T1_init, 350.0], abstol = 1.0e-10, sensealg = ForwardDiffSensitivity())
prob2 = ODEProblem(stage2!, u0, (0.0, 1.0), [T2_init], abstol = 1.0e-10, sensealg = ForwardDiffSensitivity())

# The temperature is a piecewise-constant control on ``P - 1 = 5`` uniform intervals of the
# first stage (normalized time), bounded to the reactor limits ``[298, 398]\,\mathrm{K}``.

P = 20
grid = collect(range(0.0, 1.0, P + 1))[1:(end - 1)]
temp = ControlParameter(
    grid, name = :temp, bounds = (298.0, 398.0), controls = fill(350.0, length(grid))
)

stage1 = SingleShootingLayer(
    prob1, Tsit5(); controls = (2 => temp,), bounds_p = ([10.0], [170.0])
)
stage2 = SingleShootingLayer(prob2, Tsit5(); bounds_p = ([10.0], [170.0]))

# The transition is created with an initial added volume of ``S = 0.1\,\mathrm{m^3}`` and
# bounded to ``[0, 0.1]``.

transition = StageTransition(switch_at_ts; p = [0.1], bounds = ([0.0], [0.1]))
chain = Chain(stage1, transition, stage2)

# ## The Optimization Problem

# We minimise ``J = -x_7 = -V_2 c_D(t_P)``, i.e. we **maximise** ``J_0``. The terminal
# concentration constraint ``c_D(t_P) \ge c_D^w`` is expressed as a path constraint on the
# final stage evaluated **only at the final value** (``t = 1``). The processing-time
# constraint ``J_2`` couples the two stage durations through a parameter constraint:
# ``T_1 + T_2 \le t_P``.

duration_constraint = (ps -> [ps.layer_1.p[1] + ps.layer_3.p[1] - t_p]) =>
    (lb = [-Inf], ub = [0.0])

optprob = OptimizationProblem(
    chain, AutoForwardDiff(), Val(:ComponentArrays);
    loss = :(-x₇), # maximise J0 = V2 * c_D(t_P) = D_amount(t_P)
    constraints = (
        (:x₄, 2) => (t = [1.0], bounds = (c_d_w, Inf)),
        #(:x₂, 1) => (t= [1.0], bounds = (0.0, 2000.0)) # constraint on stage 1 (not needed)
    ),
    parameter_constraints = duration_constraint,
)

uopt = solve(
    optprob, Ipopt.Optimizer();
    hessian_approximation = "limited-memory",
    max_iter = 500, print_level = 5,
);

# ## Results

# The optimized value ``J_0 = V_2 c_D(t_P) = 25.51\,\mathrm{mol}``, the switching time
# ``t_s = T_1 \approx 106\,\mathrm{min}`` and the added volume ``S \approx 0.070\,\mathrm{m^3}``
# all match the paper (``J_0 = 25.54``, ``t_s = 106.04``, ``S = 0.0702`` for ``P = 6``).
# The final concentration constraint ``c_D(t_P) = 150\,\mathrm{mol\,m^{-3}}`` is active.

ps, st = LuxCore.setup(Random.default_rng(), chain)
ps_opt = uopt + zero(ComponentArray(ps))

println("J0 = V2*c_D(t_P) = ", -uopt.objective)
println("t_s = T1 = ", ps_opt.layer_1.p[1], " min,  T2 = ", ps_opt.layer_3.p[1], " min")
println("S = ", ps_opt.layer_2.p[1], " m^3")
println("c_D(t_P) = ", last((chain(nothing, ps_opt, st))[1].u)[4], " mol/m^3")

# ### Simulating and plotting

# The trajectories are sparsely sampled for efficiency. However, this is not useful for plotting.
# We remake the layers with more densely sampled trajectories.

stage1 = SingleShootingLayer(
    remake(prob1, saveat = 0.001), Tsit5(); controls = (2 => temp,), bounds_p = ([10.0], [170.0])
)
stage2 = SingleShootingLayer(remake(prob2, saveat = 0.001), Tsit5(); bounds_p = ([10.0], [170.0]))

chain = Chain(stage1, transition, stage2)
traj, _ = chain(nothing, ps_opt, st)
traj_stage1, _ = stage1(nothing, ps_opt.layer_1, st.layer_1)

# Map the normalized time of each stage back to physical time.
T1 = ps_opt.layer_1.p[1]
T2 = ps_opt.layer_3.p[1]
t1p = T1 .* traj_stage1.t
t2p = T1 .+ T2 .* traj.t
u1 = reduce(hcat, traj_stage1.u)
u2 = reduce(hcat, traj.u)

temperature = traj_stage1[:temp]

function plot_reactor(u1, t1p, u2, t2p, temperature)
    f = Figure(size = (1000, 750))
    c = CairoMakie.Makie.wong_colors()
    ax1 = CairoMakie.Axis(
        f[1, 1], title = "A, B, C",
        xlabel = "t [min]", ylabel = "c [mol/m³]",
        xticks = 0:30:180, yticks = 0:250:2000
    )
    for (i, n) in enumerate(("A", "B", "C"))
        lines!(ax1, t1p, u1[i, :]; linewidth = 2, label = n, color = c[i])
    end
    vlines!(ax1, T1, color = :black, linestyle = :dot)
    for (i, n) in enumerate(("A", "B", "C"))
        lines!(ax1, t2p, u2[i, :]; linewidth = 2, color = c[i])
    end
    axislegend(ax1; position = :rt)

    ax2 = CairoMakie.Axis(
        f[1, 2], title = "D, E, F",
        xlabel = "t [min]", ylabel = "c [mol/m³]",
        xticks = 0:30:180, yticks = 0:50:400
    )
    for (i, n) in enumerate(("D", "E", "F"))
        lines!(ax2, t2p, u2[3 + i, :]; linewidth = 2, label = n, color = c[3 + i])
    end
    for (i, n) in enumerate(("D", "E", "F"))
        lines!(ax2, t1p, u1[3 + i, :]; linewidth = 2, color = c[3 + i])
    end
    vlines!(ax2, T1, color = :black, linestyle = :dot)
    axislegend(ax2; position = :lt)

    axc = CairoMakie.Axis(
        f[2, 1:2], title = "Optimal temperature control",
        xlabel = "t [min]", ylabel = "T [K]",
        xticks = 0:30:180,
        limits = (nothing, (305, 345))
    )
    stairs!(axc, t1p, temperature; linewidth = 2, step = :pre, label = "T(t)")
    stairs!(axc, [extrema(t2p)...], ones(2) * last(temperature); linewidth = 2, step = :pre)
    vlines!(axc, T1, color = :black, linestyle = :dot)
    axislegend(axc; position = :rt)

    return f
end

plot_reactor(u1, t1p, u2, t2p, temperature)
