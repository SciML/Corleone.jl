#src ---
#src title: Joint Lotka–Volterra experiments
#src description: Combine information from independent experimental designs.
#src ---

# ## [Multiple experiments](@id lotka_multiexp)
# Two experiments have independent fishing and measurement schedules. Their
# information matrices add in a common physical-parameter coordinate system.
using CorleoneOED, LuxCore, Random, OrdinaryDiffEqTsit5
using Optimization, OptimizationMOI, Ipopt, ComponentArrays, CairoMakie

function lotka_dynamics(x, p, t)
    return [
        x[1] - p[2] * x[1] * x[2] - 0.4 * p[1] * x[1],
        -x[2] + p[3] * x[1] * x[2] - 0.2 * p[1] * x[2],
    ]
end
problem = ODEProblem(lotka_dynamics, [0.5, 0.7], (0.0, 12.0), [0.0, 1.0, 1.0])
control = Tunable(1; default = 0.0, lowerbound = 0.0, upperbound = 1.0) => collect(0.25:0.25:11.75)
measurements = (
    ContinuousMeasurement(:w1, collect(0.0:0.25:11.75), (x, p, t) -> x[1]),
    ContinuousMeasurement(:w2, collect(0.0:0.25:11.75), (x, p, t) -> x[2]),
)
oed = OEDLayer(problem, Tsit5(), control; params = [2, 3], measurements, controls = [1])
multi = MultiExperimentLayer(oed, 2)
ps, st = LuxCore.setup(Xoshiro(1), multi)

# ## Optimize the joint information
# Budgets follow experiment order, with continuous measurements before discrete
# measurements within each experiment. Here each experiment has two densities.
optprob = OptimizationProblem(multi, ACriterion(); ps, st, M = fill(4.0, 4))
result = solve(
    optprob, Ipopt.Optimizer(); tol = 1.0e-6,
    hessian_approximation = "limited-memory", max_iter = 100, print_level = 0
)
_, reconstruct = pack(default_vectorizer(), ps)
optimal_ps = reconstruct(result.u)
trajectories, _ = multi(nothing, optimal_ps, st)
figures = map(trajectoryplot, trajectories)
F, _ = fisher_information(multi, nothing, optimal_ps, st)

# A global prior is added once, rather than once per experiment.
next_st = update_fim(multi, [(; ps = optimal_ps, st)], st)
next_st.F_init

# ## Different estimated parameter subsets
# Experiments may target different subsets. Indices must refer to the same
# physical quantities in both models. Each local matrix is embedded before
# summation; missing parameter directions contribute zero information.
first_design = OEDLayer(problem, Tsit5(), control; params = [2], measurements, controls = [1])
second_design = OEDLayer(problem, Tsit5(), control; params = [3], measurements, controls = [1])
split_design = MultiExperimentLayer((first_design, second_design))
split_ps, split_st = LuxCore.setup(Xoshiro(2), split_design)
split_problem = OptimizationProblem(
    split_design, ACriterion();
    ps = split_ps, st = split_st, M = fill(4.0, 4)
)
