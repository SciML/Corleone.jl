# ---
# title: Linear quadratic regulator
# description: Control a linear system with a tracking cost and a control penalty
# id: linear_quad_reg
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# This tutorial walks through a complete optimal-control workflow
# in Corleone: define the dynamics, declare which parameters are controls and
# which are fixed, simulate, optimize, and inspect the result.
#
# We steer two linear states with one shared input. The cost penalizes the first
# state's deviation from three and the control effort:
#
# ```math
# \begin{aligned}
# \min_u & \int_0^{12} \bigl(10 (x_1 - 3)^2 + 0.1 u^2\bigr) \, dt \\
# \text{s.t.} \quad & \dot{x}_i = a x_i + b u, \quad i = 1, 2, \\
# & x_1(0)=1,\quad x_2(0)=2, \quad a = -1, \quad b = 1.
# \end{aligned}
# ```
#
# !!! info "Prerequisites"
#     You only need the [Getting Started](../getting_started.md) guide and some
#     familiarity with ODE models in SciML.

# ## Setup

using Corleone

using LuxCore
using Random
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationLBFGSB
using ForwardDiff
using ComponentArrays
using SymbolicIndexingInterface
using CairoMakie
Makie.inline!()

# ## LQR Dynamics

# The first two states follow the linear dynamics. The third state accumulates
# the running cost, which turns the integral objective into a terminal value.
# All model quantities live in the parameter vector `p = [a, b, u]`, so the
# right-hand side never reads global variables.

function lqr_dynamics(x, p, t)
    a, b, u = p
    du = a .* x[1:(end - 1)] .+ b .* u
    costs = 10.0 .* (x[1] .- 3.0) .^ 2 .+ 0.1 .* u .^ 2
    return vcat(du, costs)
end

tspan = (0.0, 12.0)
u0 = [1.0, 2.0, 0.0]
p0 = [-1.0, 1.0, 0.0]
lqr_problem = ODEProblem(lqr_dynamics, u0, tspan, p0)

# ## [Optimal Control Problem](@id control_lqr)

# Annotations tell Corleone how to treat each parameter:
#
# - `Tunable(3; ...) => control_grid` makes `u` a piecewise-constant control.
#   The 119 switching times define 120 control intervals, and the initial values
#   are sampled uniformly from `[0, 1)`.
# - `Constant(1) => nothing` and `Constant(2) => nothing` keep `a` and `b` fixed
#   at their values from `p0`. `nothing` means "no switching times".

control_grid = collect(0.1:0.1:11.9)
predictor = ShootingLayer(
    lqr_problem, Tsit5(),
    Tunable(3; kind = Val(:parameter), default = (rng, n) -> rand(rng, n)) => control_grid,
    Constant(1; kind = Val(:parameter)) => nothing,
    Constant(2; kind = Val(:parameter)) => nothing;
    controls = [3], quadratures = [3],
)

# !!! tip "Tunable vs. Constant"
#     `Tunable` values are stored in the Lux parameters and become optimization
#     variables. `Constant` values are stored in the Lux state: they enter the
#     dynamics exactly like tunable ones but are never changed by the optimizer.

# Preview the trajectory with the initial control guess.

ps, st = LuxCore.setup(MersenneTwister(42), predictor)
solution, _ = predictor(nothing, ps, st)
trajectoryplot(solution)

# ## Optimize

# The objective is the accumulated cost `u₃` at the final time. We reuse the
# seeded parameters and state from the preview as the initial guess.

dynprob = CorleoneDynamicOptProblem(
    predictor, :(u₃(12.0)); ps, st
)

# `OptimizationProblem` packs the parameters into a flat vector and uses
# ForwardDiff for gradients. We solve with
# [`OptimizationLBFGSB.LBFGSB`](@extref Optimization).
opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
usol = solve(opt, OptimizationLBFGSB.LBFGSB())
@assert successful_retcode(usol)
(; retcode = usol.retcode, objective = usol.objective)

# Reconstruct the structured parameters from the optimizer's vector and plot the
# optimized trajectory.
_, reconstruct = pack(default_vectorizer(), dynprob.ps)
optimized_parameters = reconstruct(usol.u)
optimized_solution, _ = predictor(nothing, optimized_parameters, opt.p)
trajectoryplot(optimized_solution)

# The first state now approaches the target of three. The second state
# follows the same input but does not enter the objective.
#
# !!! note "No control bounds here"
#     The control is unbounded in this tutorial: the random initial guesses in
#     `[0, 1)` are only a starting point. Add `lowerbound` and `upperbound` to the
#     `Tunable` annotation to restrict it, as in the
#     [Lotka–Volterra fishing problem](@ref lotka_fishing).
