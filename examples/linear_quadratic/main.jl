#src ---
#src title: The Linear Quadratic Regulator
#src description: Control a linear system with a tracking cost and a control penalty
#src tags:
#src   - Beginner
#src   - Optimal Control
#src   - Linear System
#src icon: 🌊
#src ---

# # [Linear Quadratic Regulator](@id linear_quad_reg)

# We control two linear states with a shared input. The cost penalizes the first
# state's deviation from three and the control effort:

# ```math
# \begin{aligned}
# \min_u & \int_0^{12} \bigl(10 (x_1 - 3)^2 + 0.1 u^2\bigr) \, dt \\
# \text{s.t.} \quad & \dot{x}_i = -x_i + u, \quad i = 1, 2, \\
# & x_1(0)=1,\quad x_2(0)=2.
# \end{aligned}
# ```

# ## Setup

# Use Corleone to define the shooting layer and optimization model.
using Corleone

# The other packages provide:

# - Layer initialization: `LuxCore` and `Random`
# - ODE integration: `OrdinaryDiffEqTsit5`
# - Optimization: `Optimization`, `OptimizationLBFGSB`, and `ComponentArrays`
# - Objective gradients: `ForwardDiff`
# - Plotting: `CairoMakie`

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

# The first two states follow the linear dynamics. The third accumulates the
# running cost, which depends on the first state and the control.

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

# Make the third parameter a piecewise constant control and keep `a` and `b` fixed.

control_grid = collect(0.1:0.1:11.9)
predictor = ShootingLayer(
    lqr_problem, Tsit5(),
    Tunable(3; kind = Val(:parameter), default = (rng, n) -> rand(rng, n)) => control_grid,
    Constant(1; kind = Val(:parameter)) => nothing,
    Constant(2; kind = Val(:parameter)) => nothing;
    controls = [3], quadratures = [3],
)
# The 119 switching times define 120 control intervals. Initial control values
# are sampled uniformly from [0, 1). The fixed parameters are stored in Lux state.

# Preview the trajectory with the initialized control.

ps, st = LuxCore.setup(MersenneTwister(42), predictor)
solution, _ = predictor(nothing, ps, st)
trajectoryplot(solution)

# ## Optimize

# Use the accumulated cost `u₃` at the final time as the objective. Retain the
# seeded parameters and state from the preview as the optimization's initial guess.

dynprob = CorleoneDynamicOptProblem(
    predictor, :(u₃(12.0)); ps, st
)

# OptimizationProblem packs the parameters and uses ForwardDiff for gradients.
opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
usol = solve(opt, OptimizationLBFGSB.LBFGSB());
@assert successful_retcode(usol)

# Reconstruct the optimized parameters and plot the trajectory.
_, reconstruct = pack(default_vectorizer(), dynprob.ps)
optimized_parameters = reconstruct(usol.u)
optimized_solution, _ = predictor(nothing, optimized_parameters, opt.p)
trajectoryplot(optimized_solution)

# The optimized first state approaches the target of three. The second state
# follows the same input but does not enter the objective. This example uses an
# unbounded piecewise-constant control; the random initial guesses are not bounds.
usol.objective
