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

# ## Switching-Time Optimization of the Lotka-Volterra Fishing Problem

# In [the Lotka-Volterra fishing problem](@ref lotka_fishing) the fishing effort
# ``u(t) \in [0, 1]`` is a **continuous** control that is discretized into piecewise-constant
# values. Here we solve the *same* problem as a **switching-time optimization**: the control
# is restricted to the two extreme values ``u \in \{0, 1\}`` (nothing is fished, or the
# maximal effort is applied) and the only decision variables are the **durations** of the
# intervals on which each value is active.
#
# This is a natural fit for the multi-stage machinery: every interval ``i`` is an independent
# [`SingleShootingLayer`](@ref) with a fixed fishing coefficient, all stages live on the
# normalized time ``\tau \in [0, 1]``, and a single tunable parameter ``T_i`` per stage scales
# its right-hand side. A [`StageTransition`](@ref) carries the physical state from one stage
# to the next. The stage durations are coupled by a **parameter constraint** so that they sum
# to the fixed horizon ``t_f``.
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

# We bake the (fixed) fishing coefficient directly into the dynamics: each stage gets its own
# right-hand side, parameterized only by its duration ``T`` and the constant ``c \in \{0, 1\}``.
# The third state accumulates the running cost ``(x_1 - 1)^2 + (x_2 - 1)^2``. A small wrapper
# type whose coefficient `C` is fixed keeps a single right-hand side for all stages.

struct LotkaFishing{C} end

function (::LotkaFishing{C})(du, u, p, t) where {C}
    T = p[1]
    x1, x2 = u[1], u[2]
    du[1] = T * (x1 - x1 * x2 - 0.4 * C * x1)
    du[2] = T * (-x2 + x1 * x2 - 0.2 * C * x2)
    du[3] = T * ((x1 - 1.0)^2 + (x2 - 1.0)^2)
    return
end

# The horizon is ``t_f = 12``, the initial state is ``(0.5, 0.7)`` and the cost starts at zero.

const TF = 12.0
const U0 = [0.5, 0.7, 0.0]

# ## The Bang-Bang Structure

# The relaxed problem (with ``u \in [0, 1]``) has the optimal value ``1.34434`` and switches
# ``0 \to 1 \to 0`` with a long interior interval where the control follows a singular arc.
# A bang-bang control can only approximate that arc by **chattering**, so a few alternating
# ``0 \to 1 \to 0 \to \dots`` stages are needed. We use ``N = 9`` alternating stages, which
# already reproduces the optimal value to within ``1.5\times 10^{-3}``.

const N = 9
const PATTERN = [isodd(i) ? 0 : 1 for i in 1:N]

# Each stage is a separate shooting layer on ``\tau \in [0, 1]`` with its own duration as the
# only tunable parameter. The ODE tolerances are kept tight: the objective is the integral of a
# smooth function over many short and long stages, and loose tolerances inject noise into the
# outer optimization that stalls Ipopt near the solution.

function make_stage(c, T0)
    prob = ODEProblem(
        LotkaFishing{c}(), U0, (0.0, 1.0), [T0]; abstol = 1.0e-12, reltol = 1.0e-10
    )
    return ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:parameter), lowerbound = 0.0, upperbound = TF) => nothing,
    )
end

# The transition carries the two physical states forward and **resets the accumulated cost** to
# zero, so each stage integrates its own cost and the total objective is the sum of the
# per-stage terminal costs.

carry_state = StageTransition(
    (traj, _) -> [last(traj.u)[1], last(traj.u)[2], 0.0]
)

# ## Composing the Stages

# The stages and transitions are threaded into a `Stages` container. The initial durations are
# set to equal splits of the horizon.

stage_layers = [make_stage(PATTERN[i], TF / N) for i in 1:N]

layers = Any[]
for i in 1:N
    push!(layers, stage_layers[i])
    i < N && push!(layers, carry_state)
end

stages = Stages(layers...)

# ## The Optimization Problem

# The objective sums the terminal cost of **every** stage, written with the per-stage loss
# syntax `(:x₃, k)`. The stage durations are coupled by a parameter constraint enforcing
# ``\sum_i T_i = t_f``; each stage's own duration ``T_i`` is the first entry of its parameter
# vector (`stage_{2i-1}.p[1]`).

stage_names = [Symbol("stage_$(2i - 1)") for i in 1:N]

duration_constraint = (
    ps -> [
        sum(getproperty(getproperty(ps, nm).parameter.parameters, :p₁).parameter.ps[1] for nm in stage_names),
    ]
) => (lb = [TF], ub = [TF])

loss = Tuple((:u₃, k) for k in 1:N)

optprob = OptimizationProblem(
    stages, AutoForwardDiff(), Val(:ComponentArrays);
    loss = loss, parameter_constraints = duration_constraint
)

uopt = solve(
    optprob, Ipopt.Optimizer();
    hessian_approximation = "limited-memory",
    max_iter = 500, tol = 1.0e-6, print_level = 5
)

# ## Results

# The optimal value ``1.34632`` is close to the relaxed optimum ``1.34434``. The optimized
# durations show the structure of the solution: a first fishing phase, a second shorter one,
# and a long final phase without fishing.

ps, st = LuxCore.setup(Random.default_rng(), stages)
ps_opt = uopt + zero(ComponentArray(ps))
durations = [
    getproperty(getproperty(ps_opt, nm).parameter.parameters, :p₁).parameter.ps[1]
        for nm in stage_names
]

println("Objective:              ", uopt.objective)
println("Relaxed optimum:        1.34434")
println("Stage durations:        ", round.(durations; digits = 4))
println("Sum of durations:       ", sum(durations))

# ### Simulating and plotting

# The shooting layer only saves the trajectory at a handful of points (the ODE solver's
# adaptive steps and the control grid), which is far too coarse to plot. Instead we take the
# **optimized durations** and re-integrate the physical-time dynamics densely. Because the
# right-hand side is scaled by the stage duration, the physical-time dynamics are simply the
# unscaled ``\dot u = f(u)`` (the ``T_i`` factor cancels). Each stage starts from the previous
# stage's final state and runs on ``[t_0, t_0 + T_i]`` with the fixed bang-bang coefficient.

function simulate(durations, pattern; saveat = 0.005)
    t_all = Float64[]
    x_all = Vector{Float64}[]
    u_all = Float64[]
    t0 = 0.0
    state = copy(U0)
    for i in eachindex(pattern)
        Ti = durations[i]
        prob = ODEProblem(LotkaFishing{pattern[i]}(), state, (t0, t0 + Ti), [1.0])
        sol = solve(prob, Tsit5(); saveat = saveat, abstol = 1.0e-11, reltol = 1.0e-11)
        for (t, u) in zip(sol.t, sol.u)
            push!(t_all, t)
            push!(x_all, u)
            push!(u_all, pattern[i])
        end
        state = sol.u[end]
        t0 += Ti
    end
    return t_all, x_all, u_all
end

t_phys, x_traj, control = simulate(durations, PATTERN)
x1 = [u[1] for u in x_traj]
x2 = [u[2] for u in x_traj]
cost = [u[3] for u in x_traj]

function plot_sto(t, x1, x2, cost, control, durations)
    f = Figure(size = (1000, 750))
    ax1 = CairoMakie.Axis(f[1, 1], title = "States", xlabel = "t", ylabel = "x")
    lines!(ax1, t, x1; linewidth = 2, label = "prey x₁")
    lines!(ax1, t, x2; linewidth = 2, label = "predator x₂")
    for s in cumsum(durations[1:(end - 1)])
        vlines!(ax1, s; color = :black, linestyle = :dot, alpha = 0.4)
    end
    axislegend(ax1; position = :rt)

    ax2 = CairoMakie.Axis(f[1, 2], title = "Control", xlabel = "t", ylabel = "u")
    stairs!(ax2, t, control; linewidth = 2, step = :post, label = "fishing u(t)")
    for s in cumsum(durations[1:(end - 1)])
        vlines!(ax2, s; color = :black, linestyle = :dot, alpha = 0.4)
    end
    axislegend(ax2; position = :rt)

    ax3 = CairoMakie.Axis(f[2, 1:2], title = "Accumulated cost", xlabel = "t", ylabel = "cost")
    lines!(ax3, t, cost; linewidth = 2)
    return f
end

plot_sto(t_phys, x1, x2, cost, control, durations)
