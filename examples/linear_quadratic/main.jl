#src ---
#src title: The Linear Quadratic Regulator
#src description: A super beginner-friendly guide to your optimal control problem
#src tags:
#src   - Beginner
#src   - Optimal Control
#src   - Linear System
#src icon: 🌊
#src ---

# ## [Linear Quadratic Regulator](@id linear_quad_reg)

# We model a [linear quadratic regulator](https://en.wikipedia.org/wiki/Linear–quadratic_regulator)
# that drives the state toward a setpoint of three while penalizing control effort:

# ```math
# \begin{aligned}
# \min_u & \int_0^T w_1 (x - x_f)^2 + w_2 u^2 dt \\
# \text{s.t.} \quad & \dot{x} = a x + b u
# \end{aligned}
# ```
# First, we load the necessary packages to model the problem.

# ## Setup
# We will use `Corleone` to define our optimal control problem.
using Corleone

# Additionally, we will need the following packages
# - [`LuxCore`]() and [`Random`]() for basic setup functions
# - [`OrdinaryDiffEqTsit5`]() as an adaptive solver for the related ODEProblem
# - [`Optimization`](), [`OptimizationLBFGSB`](), and [`ComponentArrays`]() to setup and solve the optimization problem
# - [`CairoMakie`]() to plot the solution

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
# Next, we set up the dynamics and the objective as a single function.

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
# To convert this into an optimal control problem, we want to apply a piecewise constant control

control_grid = collect(0.1:0.1:11.9)
predictor = ShootingLayer(
    lqr_problem, Tsit5(),
    Tunable(3; kind = Val(:parameter), default = (rng, n) -> rand(rng, n)) => control_grid,
    Constant(1; kind = Val(:parameter)) => nothing,
    Constant(2; kind = Val(:parameter)) => nothing;
    controls = [3], quadratures = [3],
)
# A grid contains switching times: 119 interior times give 120 control values.
# Constants live in Lux state and do not become optimization variables.

# Lets solve the problem to have a look at its initial behavior.

ps, st = LuxCore.setup(Random.default_rng(), predictor)
solution, _ = predictor(nothing, ps, st)
trajectoryplot(solution)

# ## Optimize
# Flatten the Lux parameters for Optimization.jl and retain their axes so each
# objective call can reconstruct the parameter layout. The second state integrates
# the running cost, so its final value is the objective.
initial_parameters = ComponentArray(ps)
parameter_axes = getaxes(initial_parameters)
function lqr_objective(q, state)
    parameters = ComponentArray(q, parameter_axes)
    trajectory, _ = predictor(nothing, parameters, state)
    return last(trajectory.u)[end]
end
optfun = OptimizationFunction(lqr_objective, AutoForwardDiff())
optprob = OptimizationProblem(
    optfun, initial_parameters, st;
    lb = fill(-Inf, length(initial_parameters)), ub = fill(Inf, length(initial_parameters))
)
# And solve it using [`OptimizationLBFGSB.LBFGSB`](@extref)
uopt = solve(
    optprob, OptimizationLBFGSB.LBFGSB()
);
# Finally, lets have a look at the optimized solution.
optimized_parameters = ComponentArray(uopt.u, parameter_axes)
optimized_solution, _ = predictor(nothing, optimized_parameters, st)
trajectoryplot(optimized_solution)
