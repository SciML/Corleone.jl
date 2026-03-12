using OptimalControlBenchmarks
using ModelingToolkit
using ModelingToolkit: inputs
using Corleone
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationMOI
using ForwardDiff
using ComponentArrays
using LuxCore, Random

function solve_with_corleone(benchmark, optimizer, constraint_grid, control_grid, shooting_grid)

    data = benchmark.make_problem(constraint_grid)

    oc_problem = data.system
    tspan = data.tspan

    # scale the control grid
    control_grid = control_grid * (last(tspan) - first(tspan))
    control_grid = (control_grid .+ first(tspan))
    # scale the shooting grid
    shooting_grid = shooting_grid * (last(tspan) - first(tspan))
    shooting_grid = (shooting_grid .+ first(tspan))

    # Extract control variable
    controls = inputs(oc_problem)

    control_map = [
        c => control_grid for c in controls
    ]
    
    dynopt = CorleoneDynamicOptProblem(
        oc_problem,
        [],
        control_map...;
        algorithm = Tsit5(),
        shooting = shooting_grid
    )

    optprob = OptimizationProblem(
        dynopt,
        AutoForwardDiff(),
        Val(:ComponentArrays)
    )

    sol = solve(
        optprob,
        optimizer,
        max_iter = 1000,
        tol = 5e-6,
        hessian_approximation = "limited-memory"
    )

    # plotting
    u_opt = ComponentVector(sol.u, optprob.f.f.ax)
    opt_traj, _ = dynopt.layer(nothing, u_opt, LuxCore.initialstates(Random.default_rng(), dynopt.layer))

    f = plot_oc_problem(opt_traj, data.num_states, data.num_controls)
    display(f)
    return sol

end