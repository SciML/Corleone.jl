#src ---
#src title: Falcon 9 Booster Return and Vertical Landing
#src description: A three-phase optimal-control problem for the boost-back, coast and powered-landing arc of a Falcon 9 first stage, solved as a chain of shooting stages joined by stage transitions, with per-stage terminal path constraints
#src tags:
#src   - Multi-stage
#src   - Optimal Control
#src   - Aerospace
#src icon: 🚀
#src ---

# ## Falcon 9 Booster Return and Vertical Landing

# This example reproduces -- in a shooting-method formulation -- the three-phase rocket
# landing problem of Garrido, *Development of an Optimisation Algorithm for the Solution
# of Multiphase Optimal Control Problems using Radau Collocation* (ISR, University of
# Coimbra, 2021, **Problem 1**, Sec. 4.2). We recover the main booster of a Falcon 9
# launcher through three phases:
#
# 1. **Boost-back burn** (``t \in [0, t_1]``): three engines, throttle ``\tfrac13 T_\text{LO}``,
#    thrust direction fixed at 180° (purely horizontal, fighting the downrange velocity).
#    The duration ``t_1 = 40.8\,\mathrm{s}`` is **fixed**.
# 2. **Coast** (``t \in [t_1, t_2]``): throttle zero (``T = 0``), the vehicle is ballistic.
#    The engine reignition time ``t_2`` is **free**.
# 3. **Powered landing** (``t \in [t_2, t_f]``): one engine, throttle ``\tfrac19 T_\text{LO}``,
#    thrust direction ``\theta(t)`` free. Both ``t_f`` and ``\theta(t)`` are **free**.
#
# The only degrees of freedom are therefore ``t_2``, ``t_f`` and the landing thrust
# direction ``\theta(t)``. We maximise the final (landed) mass ``m(t_f)`` subject to a
# soft vertical touchdown at the origin:
#
# ```math
# x(t_f) = 0,\quad y(t_f) = 0,\quad v_x(t_f) = 0,\quad -0.5 \le v_y(t_f) \le 0.5 \ \mathrm{m\,s^{-1}}.
# ```
#
# Each phase lives on normalized time ``\tau \in [0,1]`` and its right-hand side is
# scaled by the (optimized) phase duration, i.e. ``\mathrm d u/\mathrm d\tau =
# T_i\, f_i(u)``. The phases are glued by identity `StageTransition`s (there are no state
# jumps for this single vehicle). The touchdown conditions are written as *path
# constraints on the third shooting stage*, which is the place where Corleone's per-stage
# constraint support shines: each constraint targets a specific stage `(:xₖ, k)`.
# The landing thrust direction is discretized on a fine grid of 100 uniform control
# intervals (``\Delta \tau = 0.01``), which lets the optimizer shape ``\theta(\tau)``
# finely enough to drive the vehicle exactly to the origin in a single NLP solve.

using Corleone
using ComponentArrays
using Ipopt
using Lux
using LuxCore
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationMOI
using Random
using SciMLBase
using SciMLSensitivity

# ## Model data (SI units)

# The vehicle flies in 2-D with state ``(x, y, v_x, v_y, m)``. Drag uses an exponential
# atmosphere ``\rho = \rho_0 \exp(-y/h_0)`` and a reference cross-section ``\pi d^2/4``.

const ρ₀ = 1.225        # kg/m³
const h₀ = 7500.0       # m
const C_D = 0.75
const d = 3.66          # m
const g₀ = 9.80665      # m/s²
const Isp = 282.0       # s
const T_LO = 5.886e6    # N (single-engine thrust of the full stack)
const T₁ = T_LO / 3     # boost-back: 3 engines
const T₃ = T_LO / 9     # landing: 1 engine
const t1 = 40.8         # s, boost-back duration (fixed)

# Initial conditions (Sec. 4.2, converted from km to m):

const x₀ = 36.022e3
const y₀ = 60.708e3
const vx₀ = 1.052e3
const vy₀ = 1.060e3
const m₀ = 76501.0

# ## Phase dynamics

# A common core computes the physical right-hand side for a thrust strength `T`, throttle
# `k` and thrust direction `(cosθ, sinθ)`. Each phase then scales it by its duration `T_dur`.

function falcon_core!(du, u, T, k, costheta, sintheta)
    x, y, vx, vy, m = u
    v = sqrt(vx^2 + vy^2)
    rho = ρ₀ * exp(-y / h₀)
    D = 0.5 * rho * C_D * (pi * d^2 / 4) * v^2   # drag, N
    du[1] = vx
    du[2] = vy
    du[3] = (T * k / m) * costheta - (D / m) * (vx / v)
    du[4] = (T * k / m) * sintheta - (D / m) * (vy / v) - g₀
    du[5] = -(T * k) / (Isp * g₀)
    return
end

function phase1!(du, u, p, t)                 # boost-back: θ = 180°, 3 engines
    T_dur = p[1]
    falcon_core!(du, u, T₁, 1.0, cos(pi), sin(pi))
    @. du = T_dur * du
    return
end
function phase2!(du, u, p, t)                 # coast: thrust 0
    T_dur = p[1]
    falcon_core!(du, u, 0.0, 0.0, 0.0, 0.0)
    @. du = T_dur * du
    return
end
function phase3!(du, u, p, t)                 # landing: θ(t) is the control, 1 engine
    T_dur, theta = p
    falcon_core!(du, u, T₃, 1.0, cos(theta), sin(theta))
    @. du = T_dur * du
    return
end

# ## Building the three-stage chain

# `make_chain` builds the `Chain` from a given initial guess ``(t_2, t_3, \{\theta\})``.
# The landing thrust direction is discretized on a fine grid of 100 uniform control
# intervals (``\Delta\tau = 0.01``), which lets the optimizer shape ``\theta(\tau)``
# finely enough to meet the (tight) terminal constraints in a single `Ipopt` solve.

grid3 = collect(0.0:0.01:0.99)   # 100 uniform control intervals on the landing stage
u0 = [x₀, y₀, vx₀, vy₀, m₀]

function make_chain(t₂, t₃, theta₀)
    prob1 = ODEProblem(phase1!, u0, (0.0, 1.0), [t1])
    prob2 = ODEProblem(phase2!, u0, (0.0, 1.0), [t₂])
    prob3 = ODEProblem(phase3!, u0, (0.0, 1.0), [t₃, theta₀[1]])
    theta = ControlParameter(grid3, name = :theta, bounds = (-pi, pi), controls = theta₀)
    stage1 = SingleShootingLayer(prob1, Tsit5(); bounds_p = ([t1], [t1]))
    stage2 = SingleShootingLayer(prob2, Tsit5(); bounds_p = ([10.0], [260.0]))
    stage3 = SingleShootingLayer(prob3, Tsit5(); controls = (2 => theta,), bounds_p = ([10.0], [260.0]))
    trans12 = StageTransition((traj, p) -> last(traj.u))   # identity: no state jump
    trans23 = StageTransition((traj, p) -> last(traj.u))
    return Chain(stage1, trans12, stage2, trans23, stage3)
end

# ## The Optimization Problem

# We solve for the free durations ``t_2`` (coast) and ``t_3`` (landing) together with the
# 100-piece control ``\theta(\tau)``, maximising the final (landed) mass
# ``m(t_f) = x_5(1)`` on the third shooting stage:
#
# ```math
# \max\; m(t_f) \quad\text{s.t.}\quad x(t_f)=0,\ y(t_f)=0,\ v_x(t_f)=0,\\ -0.5\le v_y(t_f)\le 0.5,\ \theta(t_f)=\pi/2.
# ```
#
# The touchdown conditions are per-stage path constraints acting on the **third** shooting
# stage, written `(:xₖ, 3) => (t=[1.0], bounds=...)`. The final entry constrains the
# **control** ``\theta`` of that stage (by its name `:theta`) to be exactly vertical (90°)
# at the final time. That vertical-thrust constraint is what pins down the physical
# landing: without it the optimizer leaves ``\theta`` free and wanders into a lower-mass
# local optimum with oscillating thrust and a sideways touchdown.

# ## Solve

# We start from a physically motivated guess: a long coast (near the paper's optimal
# ~180 s), a short landing burn, and a thrust direction that tilts from the boost-back
# direction toward vertical. With the fine 100-control grid and the vertical-thrust
# terminal constraint, a **single** `Ipopt` solve converges directly to the touchdown.

t₂g = 200.0
t₃g = 85.0
theta_g = collect(range(deg2rad(150.0), deg2rad(85.0), length(grid3)))
chain = make_chain(t₂g, t₃g, theta_g)

optprob = OptimizationProblem(
    chain, AutoForwardDiff(), Val(:ComponentArrays);
    loss = :(-x₅),   # maximise m(t_f) = state 5 at the end of stage 3
    constraints = (
        (:x₁, 3) => (t = [1.0], bounds = (-0.0, 0.0)),                 # downrange x = 0
        (:x₂, 3) => (t = [1.0], bounds = (-0.0, 0.0)),                 # altitude y = 0
        (:x₃, 3) => (t = [1.0], bounds = (-0.0, 0.0)),                 # horizontal velocity
        (:x₄, 3) => (t = [1.0], bounds = (-0.5, 0.5)),                 # soft vertical touchdown
        (:theta, 3) => (t = [1.0], bounds = (pi / 2 - 1.0e-3, pi / 2 + 1.0e-3)),  # vertical thrust
    ),
)

sol = solve(
    optprob, Ipopt.Optimizer(), max_iter = 2000, tol = 1.0e-9,
    constr_viol_tol = 1.0e-8, 
    hessian_approximation = "limited-memory",
    print_level = 0
)

init_ca = ComponentArray(LuxCore.initialparameters(Random.default_rng(), chain))
ps_opt = ComponentArray(sol.u, getaxes(init_ca))
println("Ipopt converged = ", SciMLBase.successful_retcode(sol))

# ## Results

# The terminal constraints are satisfied at the end of the third stage: ``x, y, v_x`` are
# pinned to 0, ``v_y \in [-0.5, 0.5]``, and the thrust is vertical at touchdown. The
# converged trajectory reproduces the reference solution of the thesis (Table 4.2) almost
# exactly -- final mass ``\approx 27.9\,\mathrm{t}``, coast ``\approx 179.7\,\mathrm{s}``,
# landing ``\approx 83.1\,\mathrm{s}``.

_, st = LuxCore.setup(Random.default_rng(), chain)
ps_ca = ps_opt  # optimized parameters as a ComponentArray
println("Final mass: ", -sol.objective, "kg")
println("coast duration  T2 = ", ps_ca.layer_3.p[1], " s")
println("landing duration T3 = ", ps_ca.layer_5.p[1], " s")
println("total time of flight tf = ", t1 + ps_ca.layer_3.p[1] + ps_ca.layer_5.p[1], " s")
println("thrust angle at touchdown theta(tf) = ", ps_ca.layer_5.controls[end], " rad")

# ### Simulating and plotting

# Remake the stages with densely sampled trajectories (`saveat`) for smooth plots. The
# shooting layers are **chained by hand** exactly as the `Chain` does internally: each
# stage is solved from the previous stage's final state (passed as its explicit `u0`),
# so the plotted trajectory is continuous across stage boundaries.

T2 = ps_ca.layer_3.p[1]
T3 = ps_ca.layer_5.p[1]
theta_ctrl = ControlParameter(
    grid3, name = :theta, bounds = (-pi, pi), controls = Vector(ps_ca.layer_5.controls)
)
prob1 = ODEProblem(phase1!, u0, (0.0, 1.0), [t1], saveat = 1e-3)
prob2 = ODEProblem(phase2!, u0, (0.0, 1.0), [T2], saveat = 1e-3)
prob3 = ODEProblem(phase3!, u0, (0.0, 1.0), [T3, ps_ca.layer_5.controls[1]], saveat = 1e-3)
stage1 = SingleShootingLayer(prob1, Tsit5(); bounds_p = ([t1], [t1]))
stage2 = SingleShootingLayer(prob2, Tsit5(); bounds_p = ([10.0], [260.0]))
stage3 = SingleShootingLayer(
    prob3, Tsit5(); controls = (2 => theta_ctrl,), bounds_p = ([10.0], [260.0])
)

traj1, _ = stage1(nothing, ps_ca.layer_1, st.layer_1)
s2start = last(traj1.u)                       # stage-1 final state feeds stage 2
traj2, _ = stage2(s2start, ps_ca.layer_3, st.layer_3)
s3start = last(traj2.u)                       # stage-2 final state feeds stage 3
traj3, _ = stage3(s3start, ps_ca.layer_5, st.layer_5)

u1 = reduce(hcat, traj1.u)
u2 = reduce(hcat, traj2.u)
u3 = reduce(hcat, traj3.u)

t_p1 = T1 .* traj1.t
t_p2 = T1 .+ T2 .* traj2.t
t_p3 = T1 .+ T2 .+ T3 .* traj3.t

using CairoMakie

fig = Figure(size = (1000, 800))
ax1 = CairoMakie.Axis(
    fig[1, 1]; title = "Booster path", xlabel = "x [km]", ylabel = "y [km]",)
# One continuous path over all three phases (absolute position, km)
lines!(ax1, u1[1, :] ./ 1e3, u1[2, :] ./ 1e3; color = :blue, label = "boost-back")
lines!(ax1, u2[1, :] ./ 1e3, u2[2, :] ./ 1e3; color = :orange, label = "coast")
lines!(ax1, u3[1, :] ./ 1e3, u3[2, :] ./ 1e3; color = :green, label = "landing")
scatter!(ax1, [u1[1, 1] / 1e3], [u1[2, 1] / 1e3]; color = :black, marker = :star5, markersize = 18, label = "start")
scatter!(ax1, [u3[1, end] / 1e3], [u3[2, end] / 1e3]; color = :red, marker = :xcross, markersize = 18, label = "landing")
axislegend(ax1; position = :lt)

ax2 = CairoMakie.Axis(fig[1, 2]; title = "Velocity", xlabel = "t [s]", ylabel = "v [m/s]")
lines!(ax2, t_p1, u1[3, :]; label = "vx", color = :red)
lines!(ax2, t_p1, u1[4, :]; label = "vy", color = :blue)
lines!(ax2, t_p2, u2[3, :]; color = :red)
lines!(ax2, t_p2, u2[4, :]; color = :blue)
lines!(ax2, t_p3, u3[3, :]; color = :red)
lines!(ax2, t_p3, u3[4, :]; color = :blue)
axislegend(ax2; position = :rt)

ax3 = CairoMakie.Axis(fig[2, 1]; title = "Mass", xlabel = "t [s]", ylabel = "m [kg]")
lines!(ax3, t_p1, u1[5, :]; color = :black)
lines!(ax3, t_p2, u2[5, :]; color = :black)
lines!(ax3, t_p3, u3[5, :]; color = :black)

ax4 = CairoMakie.Axis(fig[2, 2]; title = "Thrust angle θ", xlabel = "t [s]", ylabel = "θ [rad]")
t3c = T1 .+ T2 .+ T3 .* grid3
lines!(ax4, [0, T1], pi * ones(2))
lines!(ax4, [T1, T1+T2], zeros(2))
lines!(ax4, t3c, Vector(ps_ca.layer_5.controls); color = :green)

for ax in [ax2, ax3, ax4]
    vlines!(ax, [T1, T1+T2], color=:gray, linestyle=:dash)
end
fig 
