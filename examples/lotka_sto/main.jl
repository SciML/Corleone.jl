#src ---
#src title: Switching Time Optimization of the Lotka Volterra Fishing Problem
#src description: Bang-bang optimal control by stage-duration optimization with the multistage machinery
#src tags:
#src   - Multi-stage
#src   - Optimal Control
#src   - Switching time
#src   - Lotka Volterra
#src icon: 🎣
#src ---

# ## Switching-Time Optimization of the Lotka–Volterra Fishing Problem

# In [the Lotka–Volterra fishing problem](@ref lotka_fishing) the fishing effort
# ``w(t) \in [0, 1]`` is a **continuous** control, discretized into piecewise-constant
# values. Here we solve the *same* problem as a **switching-time optimization**: the
# control only takes the extreme values ``w \in \{0, 1\}`` (no fishing, or maximal
# effort), and the decision variables are the **durations** of the intervals on
# which each value is active.
#
# This is a natural fit for the multistage machinery: every interval ``i`` is an
# independent [`ShootingLayer`](@ref) with a fixed fishing value, all stages live
# on the normalized time ``\tau \in [0, 1]``, and a single tunable duration ``T_i``
# per stage scales its right-hand side. A [`StageTransition`](@ref) carries the
# physical state from one stage to the next, and a **parameter constraint** makes
# the durations sum to the fixed horizon ``t_f``:
#
# ```math
# \min_{T_1, \dots, T_N} \; \int_0^{t_f} (x_1 - 1)^2 + (x_2 - 1)^2 \,\mathrm{d}t
# \quad\text{s.t.}\quad \sum_i T_i = t_f, \; T_i \ge 0 .
# ```

# ## Setup

using Corleone
using LuxCore
using Random
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using CairoMakie

# ## Model

# One right-hand side serves all stages. Its parameter vector is
# `p = [T, w, c₁, c₂]`: the stage duration, the fixed fishing value of the stage,
# and the catch coefficients `c₁ = 0.4` and `c₂ = 0.2`. The third state
# accumulates the running cost ``(x_1 - 1)^2 + (x_2 - 1)^2``.

function lotka_fishing!(du, u, p, t)
    T, w, c₁, c₂ = p
    x1, x2 = u[1], u[2]
    du[1] = T * (x1 - x1 * x2 - c₁ * w * x1)
    du[2] = T * (-x2 + x1 * x2 - c₂ * w * x2)
    du[3] = T * ((x1 - 1.0)^2 + (x2 - 1.0)^2)
    return nothing
end

# The horizon is ``t_f = 12``, the initial state is ``(0.5, 0.7)``, and the cost
# starts at zero.

tf = 12.0
u0 = [0.5, 0.7, 0.0]
catch_coefficients = [0.4, 0.2]

# ## The Bang-Bang Structure

# The relaxed problem (``w \in [0, 1]``) has the optimal value ``1.34434`` and
# switches ``0 \to 1 \to 0`` with a long interior singular arc. A bang-bang control
# can only approximate that arc by **chattering**, so a few alternating stages are
# needed. ``N = 9`` alternating stages already reproduce the optimal value to
# within ``1.5\times 10^{-3}``.

N = 9
pattern = [isodd(i) ? 0.0 : 1.0 for i in 1:N]

# Each stage is a separate shooting layer on ``\tau \in [0, 1]``. Its duration is
# the only tunable parameter; the fishing value and the catch coefficients are
# `Constant`.
#
# !!! warning "Keep the ODE tolerances tight"
#     The objective sums integrals over many short and long stages. Loose
#     tolerances inject noise into the outer optimization and can stall Ipopt
#     near the solution.

function make_stage(w, T0)
    prob = ODEProblem(
        lotka_fishing!, u0, (0.0, 1.0), [T0, w, catch_coefficients...];
        abstol = 1.0e-12, reltol = 1.0e-10
    )
    return ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:parameter), lowerbound = 0.0, upperbound = tf) => nothing,
        Constant([2, 3, 4]; kind = Val(:parameter)) => nothing,
    )
end

# The transition carries the two physical states forward and **resets the
# accumulated cost** to zero, so each stage integrates its own cost and the total
# objective is the sum of the per-stage terminal costs.

carry_state = StageTransition(
    (traj, _) -> [last(traj.u)[1], last(traj.u)[2], 0.0]
)

# ## Composing the Stages

# Thread the stages and transitions into a `Stages` container, starting from
# equal splits of the horizon.

stage_layers = [make_stage(pattern[i], tf / N) for i in 1:N]

layers = Any[]
for i in 1:N
    push!(layers, stage_layers[i])
    i < N && push!(layers, carry_state)
end

stages = Stages(layers...)

# ## The Optimization Problem

# The objective sums the terminal cost of **every** stage with the per-stage loss
# syntax `(:u₃, k)`. The parameter constraint enforces ``\sum_i T_i = t_f``; the
# shooting layers are the odd entries `stage_1`, `stage_3`, … of the container.

stage_names = [Symbol("stage_$(2i - 1)") for i in 1:N]
function stage_duration(ps, name)
    group = getproperty(ps, name).parameter.parameters.p₁
    return group.parameter.p₁.ps[1]
end

# !!! note "Parameters on a shared grid are grouped"
#     The duration and the constants all use the grid `nothing`, so Corleone stores
#     them in one group that shares a single time lookup. The group is named after
#     its first member (`p₁`), and the duration is entry `p₁` inside it. Constant
#     members hold no tunable values; they live in the Lux state.

duration_constraint = (
    ps -> [sum(stage_duration(ps, name) for name in stage_names)]
) => (lb = [tf], ub = [tf])

loss = Tuple((:u₃, k) for k in 1:N)

optprob = OptimizationProblem(
    stages, AutoForwardDiff(), Val(:ComponentArrays);
    loss = loss, parameter_constraints = duration_constraint
)

uopt = solve(
    optprob, Ipopt.Optimizer();
    hessian_approximation = "limited-memory",
    max_iter = 500, tol = 1.0e-6, print_level = 0
)

# ## Results

# The optimal value ``1.34632`` is close to the relaxed optimum ``1.34434``. The
# durations reveal the structure of the solution: a first fishing phase, a second
# shorter one, and a long final phase without fishing.

ps, st = LuxCore.setup(Random.default_rng(), stages)
ps_opt = uopt + zero(ComponentArray(ps))
durations = [stage_duration(ps_opt, name) for name in stage_names]
(;
    objective = uopt.objective,
    relaxed_optimum = 1.34434,
    durations = round.(durations; digits = 4),
    total_duration = sum(durations),
)

# ### Simulating and Plotting

# The shooting layers only save a handful of points, which is too coarse to plot.
# Instead, we re-integrate the dynamics densely with the **optimized durations**.
# In physical time the duration factor cancels, so each stage runs with ``T = 1``
# on ``[t_0, t_0 + T_i]``, starting from the previous stage's final state.

function simulate(durations, pattern; saveat = 0.005)
    t_all = Float64[]
    x_all = Vector{Float64}[]
    w_all = Float64[]
    t0 = 0.0
    state = copy(u0)
    for i in eachindex(pattern)
        Ti = durations[i]
        prob = ODEProblem(
            lotka_fishing!, state, (t0, t0 + Ti), [1.0, pattern[i], catch_coefficients...]
        )
        sol = solve(prob, Tsit5(); saveat = saveat, abstol = 1.0e-11, reltol = 1.0e-11)
        for (t, u) in zip(sol.t, sol.u)
            push!(t_all, t)
            push!(x_all, u)
            push!(w_all, pattern[i])
        end
        state = sol.u[end]
        t0 += Ti
    end
    return t_all, x_all, w_all
end

t_phys, x_traj, fishing = simulate(durations, pattern)
x1 = [u[1] for u in x_traj]
x2 = [u[2] for u in x_traj]
cost = [u[3] for u in x_traj]

function plot_sto(t, x1, x2, cost, fishing, durations)
    switches = cumsum(durations[1:(end - 1)])
    f = Figure(size = (1000, 750))
    ax1 = CairoMakie.Axis(f[1, 1], title = "States", xlabel = "t", ylabel = "x")
    lines!(ax1, t, x1; linewidth = 2, label = "prey x₁")
    lines!(ax1, t, x2; linewidth = 2, label = "predator x₂")
    vlines!(ax1, switches; color = :black, linestyle = :dot, alpha = 0.4)
    axislegend(ax1; position = :rt)

    ax2 = CairoMakie.Axis(f[1, 2], title = "Control", xlabel = "t", ylabel = "w")
    stairs!(ax2, t, fishing; linewidth = 2, step = :post, label = "fishing w(t)")
    vlines!(ax2, switches; color = :black, linestyle = :dot, alpha = 0.4)
    axislegend(ax2; position = :rt)

    ax3 = CairoMakie.Axis(
        f[2, 1:2], title = "Accumulated cost", xlabel = "t", ylabel = "cost"
    )
    lines!(ax3, t, cost; linewidth = 2)
    return f
end

plot_sto(t_phys, x1, x2, cost, fishing, durations)
