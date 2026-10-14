#src ---
#src title: Manual Single Shooting with CorleoneBase
#src description: Explicit piecewise-constant fishing controls, sequential ODE stages, and a bounded objective
#src tags:
#src   - CorleoneBase
#src   - Optimal Control
#src icon: 🎣
#src ---

# # [Manual single shooting with CorleoneBase](@id base_fishing)
# We use the same relaxed [Lotka–Volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem)
# as the layer-based tutorial, but construct every stage, the objective, bounds,
# and `OptimizationProblem` ourselves. No Corleone shooting layer is imported.
# The control is a real fishing intensity in [0, 1], not a binary decision.
# This is single shooting: only controls are optimization variables; states
# are propagated, never optimized independently or matched with constraints.

using CorleoneBase
using OrdinaryDiffEqTsit5
using ForwardDiff
using Optimization
using OptimizationMOI
using Ipopt

# ## Dynamics and discretization
# With prey x, predator y, and cumulative cost z, the equations are
# x′ = x - xy - 0.4cx, y′ = -y + xy - 0.2cy,
# z′ = (x - 1)² + (y - 1)². Minimize z(12) from (0.5, 0.7, 0).
# We use 120 equal intervals of length 0.1. Control c[i] is constant on
# [grid[i], grid[i+1]); the last interval includes the final endpoint.

function fishing_dynamics!(du, u, c, t)
    x, y, z = u
    du[1] = x - x * y - 0.4 * c * x
    du[2] = -y + x * y - 0.2 * c * y
    du[3] = (x - 1)^2 + (y - 1)^2
    return nothing
end

grid = collect(0.0:0.1:12.0)
tspans = [(grid[i], grid[i+1]) for i in 1:length(grid)-1]
controls = fill(0.5, length(tspans))
initial_state = [0.5, 0.7, 0.0]

# ## Sequential stages and explicit state propagation
# `transition(seq, sol, i)` receives the sequence, the previous solution, and
# stage index: stage one receives `sol = nothing`, and later calls receive
# the preceding stored output. The control vector travels as the
# sequence's sequence-wide `parameters` field; `seq.parameters[i]` is the
# control of the interval being built. Propagate all three states, including
# accumulated cost; resetting z would minimize only the last interval.
# The sequence skeleton is constructed once below. Each evaluation remakes its
# sequence-wide parameters and states; the transition constructs every stage,
# including stage one, without mutating the skeleton. The output hook is identity.

fishing_sequence = SequentialProblem(
    ODEProblem(fishing_dynamics!, initial_state, (grid[1], grid[end]), 0.0);
    transition = function (seq, sol, i)
        prob = CorleoneBase.get_problem(seq)
        remake(
            prob;
            u0 = i == 1 ? seq.problem.u0 : sol.u[end], 
            p = seq.parameters[i],
            tspan = seq.states.tspans[i],
        )
    end,
    output = function (seq, sol, i)
        sol
    end,
    terminal = (seq, sol, i) -> i >= length(seq.states.tspans), 
    parameters = controls, 
    states = (; tspans)
)


fishing_stages = let fishing_sequence = fishing_sequence
(controls, states = fishing_sequence.states) -> begin
    stages = solve(
        fishing_sequence, Tsit5();
        parameters = controls, 
        states = states,
    )
    return stages
end
end 


# ## Objective, feasible start, and bounds
# The third state already integrates the running cost, so we read the last
# endpoint. Bounds are explicit vectors, not supplied by a control/layer API.
# Constant intensity 0.5 is the stated feasible starting control.

fishing_objective = let stages = fishing_stages 
    (controls, states) -> stages(controls, states)[end].u[end][3]
end

fishing_objective(controls, fishing_sequence.states)

lower_bounds = zero(controls)
upper_bounds = zero(controls) .+ 1.
starting_objective = fishing_objective(controls, fishing_sequence.states)
objective = OptimizationFunction(fishing_objective, AutoForwardDiff())
optproblem = OptimizationProblem(
    objective, controls, fishing_sequence.states; lb = lower_bounds, ub = upper_bounds
)

# ## Local bounded optimization
# Ipopt uses ForwardDiff gradients and a limited-memory Hessian approximation.
# Request tol=1e-6 and at most 300 iterations, using ForwardDiff for all
# 120 control variables. Disable bound relaxation so the
# physical [0, 1] bounds are not intentionally expanded by the optimizer.

optimum = solve(
    optproblem, Ipopt.Optimizer(); tol = 1.0e-6, max_iter = 300,
    hessian_approximation = "limited-memory", bound_relax_factor = 0.0,
    print_level = 0
)

optimized_control = optimum.u
optimized_stages = fishing_stages(optimized_control)
optimized_objective = optimized_stages[end].u[end][3]
