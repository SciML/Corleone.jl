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
using CommonSolve: solve
using SciMLBase: ODEProblem, remake, successful_retcode
using OrdinaryDiffEqTsit5: Tsit5
using ForwardDiff
using Optimization: OptimizationFunction, OptimizationProblem, AutoForwardDiff
using OptimizationMOI
using Ipopt

# ## Dynamics and discretization
# With prey x, predator y, and cumulative cost z, the equations are
# x′ = x - xy - 0.4cx, y′ = -y + xy - 0.2cy,
# z′ = (x - 1)² + (y - 1)². Minimize z(12) from (0.5, 0.7, 0).
# We use 24 equal intervals of length 0.5, a deliberately coarser control
# discretization than the layer example. Control c[i] is constant on
# [grid[i], grid[i+1]); the last interval includes the final endpoint.

function fishing_dynamics!(du, u, c, t)
    x, y, z = u
    du[1] = x - x * y - 0.4 * c * x
    du[2] = -y + x * y - 0.2 * c * y
    du[3] = (x - 1)^2 + (y - 1)^2
    return nothing
end

grid = collect(range(0.0, 12.0; length = 25))
ncontrols = length(grid) - 1
initial_state = [0.5, 0.7, 0.0]
ode_abstol = 1.0e-9
ode_reltol = 1.0e-9

# ## Sequential stages and explicit state propagation
# `transition(sol, i)` receives the NEXT stage index, starting at 2.
# Propagate all three states, including accumulated cost; resetting z would
# minimize only the last interval. Each fresh solve starts from initial_state.
# Promoting the first state to the control's element type supports ForwardDiff.

function fishing_stages(controls; abstol = ode_abstol, reltol = ode_reltol)
    length(controls) == ncontrols || throw(DimensionMismatch("one control per interval"))
    u0 = initial_state .+ zero(controls[1])
    first_stage = ODEProblem(
        fishing_dynamics!, u0, (grid[1], grid[2]), controls[1]
    )
    sequence = SequentialProblem(
        first_stage;
        transition = (sol, i) -> remake(
            sol.prob; u0 = sol.u[end], p = controls[i],
            tspan = (grid[i], grid[i + 1])
        ),
        terminal = (sol, i) -> i >= ncontrols
    )
    stages = solve(
        sequence, Tsit5(); abstol, reltol,
        save_everystep = false, save_start = true, save_end = true
    )
    ## Do not optimize a truncated trajectory or silently accept a failed stage.
    length(stages) == ncontrols && all(successful_retcode, stages) ||
        error("Fishing stage solve failed: $(CorleoneBase.retcode(stages))")
    return stages
end

# ## Objective, feasible start, and bounds
# The third state already integrates the running cost, so we read the last
# endpoint. Bounds are explicit vectors, not supplied by a control/layer API.
# Constant intensity 0.5 is the stated feasible starting control.

fishing_objective(controls, _) = fishing_stages(controls)[end].u[end][3]
starting_control = fill(0.5, ncontrols)
lower_bounds = zeros(ncontrols)
upper_bounds = ones(ncontrols)
starting_objective = fishing_objective(starting_control, nothing)
objective = OptimizationFunction(fishing_objective, AutoForwardDiff())
optproblem = OptimizationProblem(
    objective, starting_control; lb = lower_bounds, ub = upper_bounds
)

# ## Local bounded optimization
# Ipopt uses ForwardDiff gradients and a limited-memory Hessian approximation.
# Request tol=1e-6 and at most 300 iterations. Disable bound relaxation so the
# physical [0, 1] bounds are not intentionally expanded by the optimizer.

optimum = solve(
    optproblem, Ipopt.Optimizer(); tol = 1.0e-6, max_iter = 300,
    hessian_approximation = "limited-memory", bound_relax_factor = 0.0,
    print_level = 0
)
optimized_control = optimum.u
optimized_stages = fishing_stages(optimized_control)
optimized_objective = optimized_stages[end].u[end][3]

# ## Reproducible checks and numerical report
# Check every stage, finite endpoints/objectives, bounds (1e-8 numerical slack),
# and improvement by at least 1e-3. Reevaluate both controls with ODE tolerances
# 1e-11; require cost agreement within absolute 1e-6, relative 1e-6.
# Solver tolerances are numerical requests, not rigorous error bounds.
# Optimizer termination is reported separately from ODE stage termination.
# A successful local solve and a better feasible point do NOT prove global
# optimality, or optimality for finer grids or binary fishing controls.

bound_tolerance = 1.0e-8
@assert successful_retcode(optimum)
@assert all(successful_retcode, optimized_stages)
@assert all(sol -> all(u -> all(isfinite, u), sol.u), optimized_stages)
@assert isfinite(starting_objective) && isfinite(optimized_objective)
@assert all(isfinite, optimized_control)
@assert all(lower_bounds .- bound_tolerance .<= optimized_control .<= upper_bounds .+ bound_tolerance)
@assert optimized_objective < starting_objective - 1.0e-3
tight_start = fishing_stages(starting_control; abstol = 1.0e-11, reltol = 1.0e-11)[end].u[end][3]
tight_optimum = fishing_stages(optimized_control; abstol = 1.0e-11, reltol = 1.0e-11)[end].u[end][3]
@assert isapprox(tight_start, starting_objective; atol = 1.0e-6, rtol = 1.0e-6)
@assert isapprox(tight_optimum, optimized_objective; atol = 1.0e-6, rtol = 1.0e-6)
@assert tight_optimum < tight_start - 1.0e-3

report = (
    starting_objective = starting_objective,
    optimized_objective = optimized_objective,
    tight_objective = tight_optimum,
    control_extrema = extrema(optimized_control),
    stage_count = length(optimized_stages),
    stage_retcodes = unique(sol.retcode for sol in optimized_stages),
    optimizer_retcode = optimum.retcode,
    ode_tolerances = (abstol = ode_abstol, reltol = ode_reltol),
    optimizer_tolerance = 1.0e-6,
    bound_tolerance = bound_tolerance,
)
report
