# ---
# title: The Lotka Volterra Fishing Problem
# description: Optimize fishing intensity in a predator-prey model
# id: lotka_fishing
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# Optimize bounded fishing intensity, then add multiple shooting and continuity
# constraints. Prerequisites: the [LQR tutorial](@ref linear_quad_reg) and the
# [shooting formulation](@ref shooting_formulation).
# This example is based on the [Lotka-Volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem).

# ## [Setup](@id lotka_oc_setup)

using Corleone

using LuxCore
using Random
using OrdinaryDiffEqTsit5
using SymbolicIndexingInterface
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using ForwardDiff
using CairoMakie
Makie.inline!()
# ## Lotka–Volterra Dynamics

# The first two states are prey and predator populations. The third accumulates
# their squared deviations from one. The second parameter is fishing intensity;
# the first and third are fixed interaction rates.

# ```math
# \begin{aligned}
# \dot{x} &= x-xy-0.4vx, & x(0)&=0.5, \\
# \dot{y} &= -y+xy-0.2vy, & y(0)&=0.7, \\
# J &= \int_0^{12}\bigl((x-1)^2+(y-1)^2\bigr)\,dt, & 0&\le v\le1.
# \end{aligned}
# ```
# Here `v` is the fishing control, stored as `p[2]` in the numerical model.

function lotka_dynamics(du, u, p, t)
    du[1] = u[1] - p[1] * prod(u[1:2]) - 0.4 * p[2] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[2] * u[2]
    return du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
end
tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [1.0, 0.0, 1.0]
prob = ODEProblem(lotka_dynamics, u0, tspan, p0; abstol = 1.0e-9, reltol = 1.0e-9)

# ## Single Shooting Approach

# Make fishing intensity tunable between zero and one, and keep the two
# interaction rates fixed. The 119 switching times define 120 control intervals.
# The initial state is taken from the ODE problem.
cgrid = collect(0.1:0.1:11.9)
control = Tunable(
    2; kind = Val(:parameter), default = 1.0,
    lowerbound = 0.0, upperbound = 1.0
) => cgrid
physical_parameters = (
    Constant([3, 1]; kind = Val(:parameter)) => nothing,
)
layer = ShootingLayer(
    prob, Tsit5(), control, physical_parameters...;
    controls = [2], quadratures = [3]
)

# Minimize the terminal accumulated cost. The model initializes parameters and
# state, then prepares the objective. OptimizationProblem applies the control bounds
# from the annotation and uses ForwardDiff for gradients.
dynprob = CorleoneDynamicOptProblem(
    layer, :(u₃(12.0))
)
opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
uopt = solve(
    opt,
    Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300,
    print_level = 3
);
@assert successful_retcode(uopt)

# Reconstruct the optimized parameters and plot the trajectory.
_, reconstruct = pack(default_vectorizer(), dynprob.ps)
optimized_parameters = reconstruct(uopt.u)
optsol, _ = layer(nothing, optimized_parameters, opt.p)
trajectoryplot(optsol)

# ## Sum Up Rounding

# The relaxed control is not admissible for the underlying mixed-integer problem,
# in which the fishing intensity is binary. Sum Up Rounding (SUR) reconstructs a
# binary control from the relaxed one: on each control interval ``i`` the binary
# value ``p_i`` is chosen such that the integral of the deviation
# ``\int_0^{t} (v - w)\,d\tau`` stays within ``0.5\,\Delta t`` at all times
# (Sager, Bock & Diehl, *Solving mixed-integer control problems by Sum Up
# Rounding*). With ``q_i`` the relaxed value and the accumulator
# ``a_i = \sum_{k<i} (q_k - p_k)\Delta t``, the recursion reads
#
# ```math
# p_i = \begin{cases} 1 & \text{if } a_i + q_i\,\Delta t \ge 0.5\,\Delta t,\\ 0 & \text{else.}\end{cases}
# ```

v_relaxed = optsol[:p₂]
Δt = 12.0 / length(v_relaxed)

function sum_up_rounding(v, Δt)
    binary = zeros(length(v))
    accumulator = 0.0
    max_deviation = 0.0
    for (i, q) in enumerate(v)
        if accumulator + q * Δt >= 0.5 * Δt
            binary[i] = 1.0
            accumulator += (q - 1.0) * Δt
        else
            accumulator += q * Δt
        end
        max_deviation = max(max_deviation, abs(accumulator))
    end
    return binary, max_deviation
end
v_binary, max_deviation = sum_up_rounding(v_relaxed, Δt)

# Simulate with the fixed binary control: keep all other parameters (interaction
# rates) and replace the control values. The bound of theorem 3.1 guarantees that
# the binary trajectory stays close to the relaxed one; the objective increases
# only moderately compared to the relaxed optimum.
binary_parameters = copy(optimized_parameters)
binary_parameters.parameter.parameters.p₂.parameter.ps .= v_binary
binsol, _ = layer(nothing, binary_parameters, opt.p)
trajectoryplot(binsol)

println("relaxed objective: ", uopt.objective)
println("binary objective: ", last(binsol.u)[3])
println("maximum integral control deviation: ", max_deviation, " <= 0.5*Δt = ", 0.5 * Δt)

# ## Multiple Shooting

# Lift all three states at interior shooting points. Matching constraints enforce
# continuity, including the accumulated cost, so its final value covers the whole
# time horizon. Full-state resets allow independent blocks with `EnsembleThreads()`.
# Start Julia with `--threads=4` to use four threads.

# Start with control intervals of length 0.25. The shooting points at 3.4 and 6.1
# lie inside intervals; 9.75 already coincides with a control switch.
shooting_points = [3.4, 6.1, 9.75]
initial_conditions = (
    Tunable(i; kind = Val(:variable), lowerbound = 0.0, upperbound = 2.0) => shooting_points for i in 1:3
)
mslayer = ShootingLayer(
    prob, Tsit5(), first(control) => collect(0.25:0.25:11.75), physical_parameters...,
    initial_conditions...; ensemblealg = EnsembleThreads(),
    controls = [2], quadratures = [3]
)

# Initialize controls to one and lifted states to the corresponding ODE initial values.
msps, msst = LuxCore.setup(MersenneTwister(42), mslayer)

# Split the control intervals at 3.4 and 6.1, duplicating their current values.
msps, msst = add_parameter_segments!(mslayer, msps, msst)

# Plot the initial segments and their reset jumps.
mssol, _ = mslayer(nothing, msps, msst);
trajectoryplot(mssol, show_legend = true, whiskerwidth = 10)

# Reuse the expanded parameters and state. State continuity and matching control
# values at inserted switches are included as equality constraints.
ms_dynprob = CorleoneDynamicOptProblem(
    mslayer, :(u₃(12.0)), ps = msps, st = msst
)

ms_opt = OptimizationProblem(ms_dynprob, AutoForwardDiff(); function_kwargs = (;))
ms_uopt = solve(
    ms_opt,
    Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 1000,
    print_level = 5
);
@assert successful_retcode(ms_uopt)

# Reconstruct the optimized parameters and plot the trajectory.
_, reconstruct_ms = pack(default_vectorizer(), ms_dynprob.ps)
optimized_parameters = reconstruct_ms(ms_uopt.u)
ms_optsol, _ = mslayer(nothing, optimized_parameters, ms_opt.p)
trajectoryplot(ms_optsol)

# Matching residuals measure feasibility of the lifted states and inserted control
# intervals. The single- and multiple-shooting problems use different control
# grids, and this multiple-shooting model also bounds its lifted states (including
# the quadrature) by two. Their objective values need not coincide.
maximum(abs, shooting_constraints(ms_optsol))
