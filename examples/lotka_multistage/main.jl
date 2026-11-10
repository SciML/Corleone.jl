#src ---
#src title: A Multistage Lotka Volterra Problem with a Third Species
#src description: Optimize fishing intensity across two stages; a third species enters at the switch
#src tags:
#src   - Lotka Volterra
#src   - Optimal Control
#src   - Multi-stage
#src icon: 🎣
#src ---

# # [Multistage Lotka–Volterra with a Third Species](@id lotka_multistage)
# This example splits the [Lotka-Volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem)
# into two stages. The first stage optimizes the classic two-species fishery on
# ``[0, 12]``; at ``t = 12`` a third species (a superpredator) is introduced, and
# the second stage optimizes the three-species system on ``[12, 24]``. It is based
# on [`lotka_fishing_optimal_control`](@ref lotka_fishing) and uses the
# [`Stages`](@ref) container with a [`StageTransition`](@ref).

# ## [Setup](@id lotka_ms_setup)

# Use Corleone to define the stages and the optimization model.
using Corleone

# The other packages provide:

# - Layer initialization: `LuxCore` and `Random`
# - ODE integration: `OrdinaryDiffEqTsit5`
# - Optimization: `Optimization`, `OptimizationMOI`, `Ipopt`, and `ComponentArrays`
# - Objective and constraint gradients: `ForwardDiff`
# - Plotting: `CairoMakie`

using LuxCore
using Random
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using ForwardDiff
using CairoMakie
Makie.inline!()
# ## Dynamics

# The first two states are prey and predator populations; the fishing control `v`
# enters both equations. The third state accumulates the squared deviations of the
# populations from one. The stage durations enter as scaling parameters, so both
# stages live on the normalized time ``\tau \in [0, 1]``.

# ```math
# \begin{aligned}
# \dot{x} &= T_1(x-xy-0.4vx), & \dot{y} &= T_1(-y+xy-0.2vy),\\
# \dot{J}_1 &= T_1\bigl((x-1)^2+(y-1)^2\bigr), & 0&\le v\le1.
# \end{aligned}
# ```

function lotka_two_species(du, u, p, t)
    x, y, cost = u
    T, v = p
    du[1] = T * (x - x * y - 0.4 * v * x)
    du[2] = T * (-y + x * y - 0.2 * v * y)
    return du[3] = T * ((x - 1.0)^2 + (y - 1.0)^2)
end

# In the second stage the superpredator ``x_3`` preys on the predator. With
# ``\dot{x}_3 = T_2 x_3 (y - x_3 - 0.3 v)`` the triple ``(1, 1, 1)`` is a steady
# state at ``v = 0``. The cost accumulator continues from the first stage and adds
# the third species' deviation.

function lotka_three_species(du, u, p, t)
    x, y, x3, cost = u
    T, v = p
    du[1] = T * (x - x * y - 0.4 * v * x)
    du[2] = T * (-y + x * y - 0.2 * v * y)
    du[3] = T * (x3 * (y - x3 - 0.3 * v))
    return du[4] = T * ((x - 1.0)^2 + (y - 1.0)^2 + (x3 - 1.0)^2)
end

const T1 = 12.0
const T2 = 12.0
const ETA = 0.5 # initial population of the third species

prob1 = ODEProblem(lotka_two_species, [0.5, 0.7, 0.0], (0.0, 1.0), [T1, 0.0]; abstol = 1.0e-10, reltol = 1.0e-9)
prob2 = ODEProblem(lotka_three_species, [0.5, 0.7, ETA, 0.0], (0.0, 1.0), [T2, 0.0]; abstol = 1.0e-10, reltol = 1.0e-9)

# ## Defining the Stages

# The stage durations are fixed parameters (the third species enters exactly at
# ``t = 12``). Each stage has its own piecewise-constant fishing control on 40
# intervals of the normalized horizon. The initial state is taken from the ODE
# problems; the third species enters through the transition at the switch.

cgrid = collect(0.01:0.01:0.99)
control(T) = (
    Constant(1; kind = Val(:parameter)) => nothing,
    Tunable(2; kind = Val(:parameter), default = 0.0, lowerbound = 0.0, upperbound = 1.0) => cgrid,
)
stage1 = ShootingLayer(prob1, Tsit5(), control(T1)...; controls = [2], quadratures = [3])
stage2 = ShootingLayer(prob2, Tsit5(), control(T2)...; controls = [2], quadratures = [4])

# At ``t = 12`` the populations and the accumulated cost are carried over and the
# third species is introduced with population ``\eta = 0.5``.

transition = StageTransition(
    (traj, _) -> begin
        xf = last(traj.u)
        [xf[1], xf[2], ETA, xf[3]]
    end
)
stages = Stages(stage1, transition, stage2)

# ## The Optimization Problem

# Minimize the terminal accumulated cost of the final stage.
optprob = OptimizationProblem(
    stages, AutoForwardDiff(), Val(:ComponentArrays);
    loss = :u₄,
)

uopt = solve(
    optprob,
    Ipopt.Optimizer(),
    tol = 1.0e-8,
    hessian_approximation = "limited-memory",
    max_iter = 1000,
    print_level = 3
);
@assert successful_retcode(uopt)

# ## Results

# The optimized cost, the superpredator's terminal population, and the fraction of
# intervals in which fishing is active in either stage. Fishing concentrates on the
# first stage: the two-species fishery is cheaper to exploit before the costly
# third species must be driven toward one.

ps, st = LuxCore.setup(Random.default_rng(), stages)
ps_opt = uopt + zero(ComponentArray(ps))
traj2, _ = stages(nothing, ps_opt, st)

v1 = Vector(ps_opt.stage_1.parameter.parameters.p₂.parameter.ps)
v2 = Vector(ps_opt.stage_3.parameter.parameters.p₂.parameter.ps)
println("objective: ", uopt.objective)
println("x3(24) = ", last(traj2.u)[3])
println("active intervals stage 1: ", count(>(0.01), v1), " / ", length(v1))
println("active intervals stage 2: ", count(>(0.01), v2), " / ", length(v2))

# ### Simulating and plotting

# The trajectories are sparsely sampled for efficiency. Remake the layers with
# densely sampled trajectories for plotting.

stage1_dense = ShootingLayer(remake(prob1, saveat = 0.001), Tsit5(), control(T1)...; controls = [2], quadratures = [3])
stage2_dense = ShootingLayer(remake(prob2, saveat = 0.001), Tsit5(), control(T2)...; controls = [2], quadratures = [4])
stages_dense = Stages(stage1_dense, transition, stage2_dense)

_, st_dense = LuxCore.setup(Random.default_rng(), stages_dense)
traj_stage1, _ = stage1_dense(nothing, ps_opt.stage_1, st_dense.stage_1)
traj2_dense, _ = stages_dense(nothing, ps_opt, st_dense)

t1 = T1 .* traj_stage1.t
t2 = T1 .+ T2 .* traj2_dense.t
u1 = reduce(hcat, traj_stage1.u)
u2 = reduce(hcat, traj2_dense.u)

# The control is piecewise constant on `cgrid`; resample it on the dense
# normalized trajectory time so it can be drawn as a stair plot.
switch_times = vcat(T1 .* cgrid, T1 .+ T2 .* cgrid)
v_series = vcat(v1, v2)
v_full = v_series[clamp.(searchsortedlast.(Ref(switch_times), vcat(t1, t2)), 1, length(v_series))]

function plot_lotka(u1, t1, u2, t2, v_full)
    f = Figure(size = (1000, 750))
    c = CairoMakie.Makie.wong_colors()
    ax1 = CairoMakie.Axis(f[1, 1], title = "Populations", xlabel = "t", ylabel = "population")
    lines!(ax1, t1, u1[1, :]; linewidth = 2, label = "x (prey)", color = c[1])
    lines!(ax1, t1, u1[2, :]; linewidth = 2, label = "y (predator)", color = c[2])
    lines!(ax1, t2, u2[1, :]; linewidth = 2, color = c[1])
    lines!(ax1, t2, u2[2, :]; linewidth = 2, color = c[2])
    lines!(ax1, t2, u2[3, :]; linewidth = 2, label = "x3 (superpredator)", color = c[3])
    vlines!(ax1, T1, color = :black, linestyle = :dot)
    axislegend(ax1; position = :rt)

    axc = CairoMakie.Axis(f[2, 1], title = "Optimal fishing control", xlabel = "t", ylabel = "v(t)")
    stairs!(axc, vcat(t1, t2), v_full; linewidth = 2, step = :pre)
    vlines!(axc, T1, color = :black, linestyle = :dot)

    axj = CairoMakie.Axis(f[2, 2], title = "Accumulated cost", xlabel = "t", ylabel = "J(t)")
    lines!(axj, t1, u1[3, :]; linewidth = 2, color = c[5])
    lines!(axj, t2, u2[4, :]; linewidth = 2, color = c[5])
    vlines!(axj, T1, color = :black, linestyle = :dot)

    return f
end

plot_lotka(u1, t1, u2, t2, v_full)
