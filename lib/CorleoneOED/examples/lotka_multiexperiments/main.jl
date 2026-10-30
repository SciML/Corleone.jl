#src ---
#src title: Joint Lotka–Volterra experiments
#src description: Combine information from independent experimental designs.
#src ---

# # [Multiple experiments](@id lotka_multiexp)
# Two experiments have independent fishing and measurement schedules. Their
# information matrices add in a common physical-parameter coordinate system.
# Both use `p = [fishing, interaction_1, interaction_2]`, estimating `[2,3]` while
# holding those rates fixed at nominal values. Fishing and both measurement
# densities are tunable. This repeats the [single-experiment model](@ref lotka_oed);
# identical nominal experiments may yield symmetric optimal designs. Independence
# of their parameter trees does not force different schedules.
#
# ComponentArrays supplies packing; Optimization/OptimizationMOI and Ipopt solve
# the relaxation; CairoMakie plots each experiment. No shooting resets are used,
# so there are no automatic shooting equalities in this example.
using CorleoneOED, LuxCore, Random, OrdinaryDiffEqTsit5, LinearAlgebra
using Optimization, OptimizationMOI, Ipopt, ComponentArrays, CairoMakie
using SymbolicIndexingInterface: variable_symbols, variable_index

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
ps, st = LuxCore.setup(Xoshiro(1), multi);

# ## Optimize the joint information
# Budgets follow experiment order, with continuous measurements before discrete
# measurements within each experiment. Here each experiment has two densities.
# Setup uses a reproducible seed; parameter and state trees are independent.
optprob = OptimizationProblem(multi, ACriterion(); ps, st, M = fill(4.0, 4))
result = solve(
    optprob, Ipopt.Optimizer(); tol = 1.0e-6,
    hessian_approximation = "limited-memory", max_iter = 100, print_level = 0
);
@assert CorleoneOED.SciMLBase.successful_retcode(result)
@assert isfinite(result.objective)
_, reconstruct = pack(default_vectorizer(), ps)
optimal_ps = reconstruct(result.u)
trajectories, _ = multi(nothing, optimal_ps, st)
F, _ = fisher_information(multi, nothing, optimal_ps, st)
optimal_budgets = sampling_sums(multi, trajectories)
@assert isposdef(Symmetric(F))
@assert all(optimal_budgets .<= 4.0 + 1.0e-5)
@assert isapprox(result.objective, ACriterion()(F); rtol = 1.0e-5)
local_information = map(values(multi.experiments), values(trajectories)) do layer, trajectory
    fisher_information(layer, trajectory; add_initial = false)
end
@assert F ≈ sum(local_information) # both experiments use the same full coordinates
(; termination = result.retcode, objective = result.objective, budgets = optimal_budgets)

# The first experiment's physical populations, fishing (`p₁`), and densities
# (`w1`, `w2`). Exclude augmented sensitivity/information states from these plots.
augmented_states = filter(variable_symbols(trajectories.experiment_1)) do symbol
    variable_index(trajectories.experiment_1, symbol) > length(problem.u0)
end
trajectoryplot(trajectories.experiment_1; skips = augmented_states, state_string = "Population")

# The second experiment's populations and independent schedules. Access experiments
# by name: iterating a ComponentArray would yield its numerical entries instead.
trajectoryplot(trajectories.experiment_2; skips = augmented_states, state_string = "Population")

# ## Carry information into another design
# A global prior is added once, rather than once per experiment. Here an identity
# prior is an illustrative input, not a statistical calibration recommendation.
prior_state = merge(st, (; F_init = Matrix{Float64}(I, 2, 2)))
F_with_prior, _ = fisher_information(multi, nothing, optimal_ps, prior_state)
@assert F_with_prior ≈ F + prior_state.F_init
next_st = update_fim(multi, [(; ps = optimal_ps, st)], prior_state)
@assert next_st.F_init ≈ prior_state.F_init + F
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
);
# This last problem is prepared but not solved. Each experiment contributes one
# parameter direction; missing directions are zero-filled during embedding. SVD
# reductions use the approximation described in [the reduction guide](@ref oed_svd).
