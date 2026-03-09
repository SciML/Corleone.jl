using ModelingToolkit
using ModelingToolkit: inputs
using Corleone
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationMOI
using Ipopt
using ForwardDiff
using ComponentArrays
using LuxCore, Random

function solve_with_corleone(benchmark)

    data = benchmark.make_problem()

    oc_problem = data.system
    cgrid = data.control_grid

    # Extract control variable
    controls = inputs(oc_problem)

    control_map = [
        c => cgrid for c in controls
    ]
    
    dynopt = CorleoneDynamicOptProblem(
        oc_problem,
        [],
        control_map...;
        algorithm = Tsit5()
    )

    optprob = OptimizationProblem(
        dynopt,
        AutoForwardDiff(),
        Val(:ComponentArrays)
    )

    sol = solve(
        optprob,
        Ipopt.Optimizer(),
        max_iter = 1000,
        tol = 5e-6,
        hessian_approximation = "limited-memory"
    )

    return sol

end