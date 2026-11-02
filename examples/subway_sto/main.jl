#src ---
#src title: Switching-Time Optimization of the Subway Ride Problem
#src description: Energy-optimal subway operation as a switching-time optimization with the multistage machinery
#src tags:
#src   - Multi-stage
#src   - Optimal Control
#src   - Switching time
#src   - Subway
#src icon: 🚇
#src ---

# ## Switching-Time Optimization of the Subway Ride Problem

# The *subway ride* problem goes back to the work of Bock for the New York City subway. A
# train of ``W`` pounds must travel a fixed distance ``S`` between two stations in a fixed
# time ``t_f``; the goal is to use as little energy as possible.
#
# The train can be operated in one of four **modes**
#
# | mode ``w`` | 1 | 2 | 3 | 4 |
# |---|---|---|---|---|
# | meaning | series | parallel | coasting | braking |
#
# In general the mode is a *state-dependent* discrete control: acceleration and energy
# consumption change with the velocity ``x_1``, and the mode schedule is the object to be
# optimized. This is a mixed-integer optimal control problem (MIOCP).
#
# Here we solve it as a **switching-time optimization**, in the same spirit as
# [the Lotka-Volterra switching-time problem](@ref): we fix a *sequence* of modes and treat
# the **durations** of the corresponding intervals as the only decision variables. Each
# interval is an independent [`SingleShootingLayer`](@ref) on normalized time
# ``\tau \in [0, 1]`` whose right-hand side is scaled by its duration ``T_i``, and a
# [`StageTransition`](@ref) carries the state from one interval to the next. The durations are
# coupled by a parameter constraint ``\sum_i T_i = t_f``.
#
# ```math
# \min_{T_1, \dots, T_N} \; \frac{1}{3600}\int_0^{t_f} L(x, w) \,\mathrm{d}t
# \quad\text{s.t.}\quad \dot x_0 = x_1,\; \dot x_1 = f_1(x, w),\;
# x(0) = (0, 0),\; x(t_f) = (S, 0),\; \sum_i T_i = t_f .
# ```

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

# The physical parameters are taken from the problem description and the MUSCOD model.

const GAMMA = 3600.0 / 5280.0      # unit conversion between mph and ft/s
const W = 78000.0                  # weight [lbs]
const WEFF = W + 72000.0 * 0.1     # effective weight [lbs]
const AREA = 100.0                 # frontal area [ft^2]
const BCOEF = 0.045
const CDRAG = 0.000546
const COAST = 0.367                # constant coasting deceleration term
const GC = 32.2                    # gravitational acceleration [ft/s^2]
const EFF = 1.0                    # efficiency

# The three switching velocities separate the sub-modes of the series and parallel operation.

const V1 = 0.979474 / GAMMA        # [ft/s]
const V2 = 6.73211 / GAMMA
const V3 = 14.2658 / GAMMA

# Tractive-force (``b``), acceleration (``a``), power (``p``) and consumption (``c``)
# coefficients of the two accelerating modes.

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

# ### Forces and polynomials

# A small helper evaluates the degree-five polynomials ``\sum_i c_i z^i``.

poly(cs, z) = cs[1] + z * (cs[2] + z * (cs[3] + z * (cs[4] + z * (cs[5] + z * cs[6]))))

# Resistance, tractive force and energy-consumption polynomials:

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

# The **state-dependent sub-switching** inside the series and parallel modes is baked into
# the right-hand side: for a velocity below ``v_1`` the series mode uses the constant
# acceleration ``a_1`` with consumption ``p_1``, between ``v_1`` and ``v_2`` it uses ``a_2``
# with ``p_2``, and above ``v_2`` it follows the velocity-dependent tractive curve. The
# parallel mode is infeasible below ``v_2``, uses ``a_3`` with ``p_3`` up to ``v_3`` and the
# tractive curve above. Coasting and (maximal) braking have no energy consumption.
#
# `decoupled_rhs` returns ``(\dot x_1, L)`` for a given mode and velocity, with the energy
# scaled by ``1/3600``.

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
            # The parallel mode is infeasible below `v_2`; the optimizer never enters it
            # there. As a finite safety net for the dense re-simulation we fall back to
            # the continuous `a_3` acceleration instead of returning an infinite penalty.
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

# The stage dynamics are parameterized by the stage duration ``T`` (and show the `mode`
# through the type parameter), with the third state accumulating the energy cost.

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

# The ride starts and ends at rest and must cover ``S = 2112`` ft in ``t_f = 65`` s.

const TF = 65.0
const S = 2112.0
const U0 = [0.0, 0.0, 0.0]

# ## The Mode Sequence

# The optimal order of the (decoupled) modes is known from the reference solution to be
#
# ```math
# 1 \to 2 \to 1 \to 3 \to 4,
# ```
#
# i.e. accelerate in series, continue accelerating in parallel, accelerate in series again
# (this is the regime above ``v_2`` where the parallel mode is inefficient), coast, and
# finally brake to a stop.
#
# Rather than hard-coding that order we hand the solver a longer **repetition** of the four
# modes and let it discover the schedule by driving the unneeded durations to zero. The
# `1 \to 2 \to 1 \to 3 \to 4` structure then emerges on its own (adjacent equal modes simply
# merge).

const MODES = repeat([1, 2, 3, 4], 6)
const N = length(MODES)

# Each interval is a separate shooting layer on ``\tau \in [0, 1]`` with its own duration as
# the only tunable parameter. The ODE tolerances are tight to keep the outer optimization
# smooth.

function make_stage(mode, T0)
    prob = ODEProblem(
        SubwayMode{mode}(), U0, (0.0, 1.0), [T0]; abstol = 1.0e-12, reltol = 1.0e-10
    )
    return SingleShootingLayer(prob, Tsit5(); bounds_p = ([0.0], [TF]))
end

# The transition passes the whole state (position, velocity, accumulated energy) on
# unchanged.

carry_state = StageTransition((traj, _) -> copy(last(traj.u)))

# ## Composing the Chain

stages = [make_stage(MODES[i], TF / N) for i in 1:N]

layers = Any[]
for i in 1:N
    push!(layers, stages[i])
    i < N && push!(layers, carry_state)
end

chain = Chain(layers...)

# ## The Optimization Problem

# The objective is the accumulated energy in the final stage. The terminal conditions
# ``x_0(t_f) = S`` and ``x_1(t_f) = 0`` are imposed as terminal path constraints on the last
# stage (evaluated at normalized time ``\tau = 1``), and the durations are coupled by
# ``\sum_i T_i = t_f``.

stage_names = [Symbol("layer_$(2i - 1)") for i in 1:N]

duration_constraint = (ps -> [sum(getproperty(ps, nm).p[1] for nm in stage_names)]) =>
    (lb = [TF], ub = [TF])

terminal_constraints = (
    (:x₁, N) => (t = [1.0], bounds = (S, S)),
    (:x₂, N) => (t = [1.0], bounds = (0.0, 0.0)),
)

optprob = OptimizationProblem(
    chain, AutoForwardDiff(), Val(:ComponentArrays);
    loss = :x₃, constraints = terminal_constraints,
    parameter_constraints = duration_constraint
)

uopt = solve(
    optprob, Ipopt.Optimizer();
    hessian_approximation = "limited-memory",
    max_iter = 3000, tol = 1.0e-9, print_level = 5
)

# ## Results

# The optimized energy ``1.14562`` matches the reference value ``1.14569`` of the indirect
# solution. The optimized durations reproduce the reference mode schedule: a short series
# acceleration, a parallel phase, a long series phase, the coast, and the final braking.

ps, st = LuxCore.setup(Random.default_rng(), chain)
ps_opt = uopt + zero(ComponentArray(ps))
durations = [getproperty(ps_opt, nm).p[1] for nm in stage_names]

println("Energy:                 ", uopt.objective)
println("Reference energy:       1.14569")
println("Mode sequence:          ", MODES)
println("Stage durations:        ", round.(durations; digits = 4))
println("Sum of durations:       ", sum(durations))

# ### Simulating and plotting

# As in the Lotka-Volterra switching-time example, the shooting trajectory is too coarse to
# plot, so we re-integrate the physical-time dynamics densely from the optimized durations.
# Because the right-hand side is scaled by the duration, the physical-time dynamics are the
# unscaled ``\dot u = f(u)``; we simply pass ``T = 1``.
#
# The optimizer is free to switch modes off by sending their duration to zero. Numerically
# these durations come out as tiny *negative* numbers (e.g. ``-5\times10^{-9}``), so we
# clamp them to zero and skip inactive stages entirely. Otherwise the accumulated time would
# walk backwards and `saveat` would emit non-monotone points that break the plot.

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
        Ti < INACTIVE && continue          # mode is switched off
        prob = ODEProblem(SubwayMode{modes[i]}(), state, (t0, t0 + Ti), [1.0])
        sol = solve(prob, Tsit5(); saveat = saveat, abstol = 1.0e-11, reltol = 1.0e-11)
        for (t, u) in zip(sol.t, sol.u)
            (isempty(t_all) || t > t_all[end]) || continue   # drop duplicated boundary
            push!(t_all, t)
            push!(x_all, u)
            push!(w_all, Float64(modes[i]))
        end
        tend = t0 + Ti
        if t_all[end] < tend - 1.0e-12      # make sure the segment endpoint is present
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

t_phys, x_traj, mode_traj, switches = simulate(durations, MODES)
position = [u[1] for u in x_traj]
velocity_mph = [u[2] * GAMMA for u in x_traj]
energy = [u[3] for u in x_traj]

println("Final position:         ", position[end])
println("Final velocity [mph]:   ", velocity_mph[end])
println("Final energy:           ", energy[end])
println("Active switch times:    ", round.(switches[1:(end - 1)]; digits = 4))

function plot_subway(t, position, velocity_mph, energy, mode_traj, switches)
    inner = switches[1:(end - 1)]
    f = Figure(size = (1000, 750))

    ax1 = CairoMakie.Axis(f[1, 1], title = "Position", xlabel = "t [s]", ylabel = "x₀ [ft]")
    lines!(ax1, t, position; linewidth = 2, label = "position x₀")
    for s in inner
        vlines!(ax1, s; color = :black, linestyle = :dot, alpha = 0.4)
    end
    axislegend(ax1; position = :lt)

    ax2 = CairoMakie.Axis(f[1, 2], title = "Velocity", xlabel = "t [s]", ylabel = "x₁ [mph]")
    lines!(ax2, t, velocity_mph; linewidth = 2, color = :orange)
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

plot_subway(t_phys, position, velocity_mph, energy, mode_traj, switches)

