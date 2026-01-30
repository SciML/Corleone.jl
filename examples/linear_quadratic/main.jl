#src ---
#src title: The Linear Quadratic Regulator
#src description: A super beginner-friendly guide to your optimal control problem
#src tags:
#src   - Beginner
#src   - Optimal Control
#src   - Linear System
#src icon: 🌊
#src ---

# Let’s start by modeling the [linear quadratic regulator](https://en.wikipedia.org/wiki/Linear–quadratic_regulator). This example is one of the simplest optimal control problems. Lets assume we want to It reads as

# \begin{align}
# \min_u & \int_0^T w_1 (x - x_f)^2 + w_2 u^2 dt \\
# \text{s.t.} \quad & \dot{x} = a x + b u
# \end{align}

# First, we load the necessary packages to model the problem.

# ## Setup
# We will use `Corleone` to define our optimal control problem.
using Corleone

# Additionally, we will need the folllowing packages
# - [`LuxCore`]() and [`Random`]() for basic setup functions
# - [`OrdinaryDiffEqTsit5`]() as an adaptive solver for the related ODEProblem
# - [`Optimization`](), [`OptimizationLBFGSB`](), and [`ComponentArrays`]() to setup and solve the optimization problem
# - [`CairoMakie`]() to plot the solution

using LuxCore
using Random
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationLBFGSB
using ComponentArrays
using CairoMakie

# ## LQR Dynamics
# Next, we set up the dynamics and the objective as a single function.

function lqr_dynamics(x, p, t)
    a, b, u = p
    du = a .* x[1] .+ b .* u
    costs = 10.0 .* (x[1] .- 3.0) .^ 2 .+ 0.1 .* u .^ 2
    return vcat(du, costs)
end

tspan = (0.0, 12.0)
u0 = [1.0, 0.0]
p0 = [-1.0, 1.0, 0.0]
lqr_problem = ODEProblem(lqr_dynamics, u0, tspan, p0)

# ## Optimal Control Problem
# To convert this into an optimal control problem, we want to apply a piecewise constant control

control_function = ControlParameter(
    0.0:0.1:11.9, name = :input
)

predictor = SingleShootingLayer(
    lqr_problem, Tsit5(),
    controls = (3 => control_function,),
    bounds_p = ([-1.0, 1.0], [-1.0, 1.0])
)

# Lets solve the problem to have a look at its initial behavior.

ps, st = LuxCore.setup(Random.default_rng(), predictor)
solution, _ = predictor(nothing, ps, st)

# We define a plotting function here
function plot_lqr_solution(solution; show_setpoint = true)
    f = Figure()
    axes = [CairoMakie.Axis(f[i, 1], xlabel = i == 3 ? "t" : "") for i in 1:3]
    plot!(axes[1], solution, vars = [:x₁])
    if show_setpoint
        hlines!(axes[1], [3.0], color = :black, linestyle = :dash)
    end
    stairs!(axes[2], solution, vars = [:u₁])
    plot!(axes[3], solution, vars = [:x₂])
    for (i, title) in enumerate(["State", "Control", "Costs"])
        f[i, 2] = Legend(f, axes[i], title, framevisible = false)
    end
    return f
end

plot_lqr_solution(solution)

# ## Optimize
# Transforming the given layer into an [`SciMLBase.OptimizationProblem`](@extref) is a one liner. We simply give in the predictor and the cost to minimize.
optprob = OptimizationProblem(
    predictor, :x₂
)
# And solve it using [`OptimizationLBFGSB.LBFGSB`](@extref)
uopt = solve(
    optprob, OptimizationLBFGSB.LBFGSB()
)
# Finally, lets have a look at the optimized solution.
optimized_solution, _ = predictor(nothing, uopt + zero(ComponentArray(ps)), st)
plot_lqr_solution(optimized_solution)

