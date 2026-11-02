#src ---
#src title: Lotka–Volterra optimal experimental design
#src description: Jointly optimize fishing and continuous measurement schedules.
#src ---

# # [Lotka–Volterra OED](@id lotka_oed)
# We design an experiment to estimate the two interaction coefficients of a
# Lotka–Volterra model. Fishing is a physical control; two measurement densities
# select when to observe each population. This follows the
# [Lotka experimental-design benchmark](https://mintoc.de/index.php?title=Lotka_Experimental_Design).

using CorleoneOED, LuxCore, Random, OrdinaryDiffEqTsit5, LinearAlgebra
using Optimization, OptimizationMOI, Ipopt, ComponentArrays, CairoMakie
using SymbolicIndexingInterface: variable_symbols, variable_index

function lotka_dynamics(du, x, p, t)
    du[1] = x[1] - p[2] * x[1] * x[2] - 0.4 * p[1] * x[1]
    du[2] = -x[2] + p[3] * x[1] * x[2] - 0.2 * p[1] * x[2]
    return nothing
end
problem = ODEProblem(lotka_dynamics, [0.5, 0.7], (0.0, 12.0), [0.0, 1.0, 1.0])

# The original parameter vector is `p = [fishing, interaction_1, interaction_2]`.
# We estimate indices `[2,3]` at their nominal values; only fishing and measurement
# weights are design variables. The [formulation](@ref oed_formulation) explains
# this distinction. No physical shooting resets are introduced in this tutorial,
# so the automatic shooting-equality block is empty.
#
# ## Define controls and measurements
# The fishing control has one value per interval, including the first interval.
# Unannotated physical parameters remain fixed at their nominal values. `params`
# selects the parameters whose sensitivities determine information.
control = Tunable(1; default = 0.0, lowerbound = 0.0, upperbound = 1.0) => collect(0.25:0.25:11.75)
measurements = (
    ContinuousMeasurement(:w1, collect(0.0:0.25:11.75), (x, p, t) -> x[1]),
    ContinuousMeasurement(:w2, collect(0.0:0.25:11.75), (x, p, t) -> x[2]),
)
oed = OEDLayer(problem, Tsit5(), control; params = [2, 3], measurements, controls = [1])
ps, st = LuxCore.setup(Xoshiro(1), oed);
trajectory, _ = oed(nothing, ps, st);
# Keep the physical states visible instead of overlaying the augmented G/F states.
augmented_states = filter(variable_symbols(trajectory)) do symbol
    variable_index(trajectory, symbol) > length(problem.u0)
end
trajectoryplot(trajectory; skips = augmented_states, state_string = "Population")

# Initial populations, fishing, and unit measurement densities. Controls and
# measurement weights are parameter histories, not appended physical states.
# The legend's `p₁` is fishing, while `w1` and `w2` are the two densities.

# The layer integrates state sensitivities `G` and continuous information `F`:
# ```math
# G' = f_xG + f_\theta, \qquad
# F' = \sum_m w_m(t)J_m(t)^\top J_m(t), \qquad J_m=h_{m,x}G.
# ```
# Here observations depend only on states, so the direct term `h_θ` is zero.
# Each weight lies in `[0,1]`. Its time integral is the measurement budget.
F = fisher_information(oed, trajectory)
budgets = sampling_sums(oed, trajectory)
(; initial_information = F, initial_budgets = budgets)

# ## Optimize the design
# Minimize the trace of inverse information with at most four time units of
# measurement for each population. The facade adds the sampling constraints after
# user constraints and before automatic shooting equalities. All grids stay fixed
# during optimization. ComponentArrays supplies the shared vectorizer; Optimization,
# OptimizationMOI, and Ipopt supply the continuous optimizer, and CairoMakie supplies
# plots. These packages are not required just to run an OED layer.
model = CorleoneDynamicOptProblem(oed, ACriterion(); ps, st, M = [4.0, 4.0])
optprob = OptimizationProblem(model, AutoForwardDiff())
result = solve(
    optprob, Ipopt.Optimizer(); tol = 1.0e-6,
    hessian_approximation = "limited-memory", max_iter = 100, print_level = 0
);
@assert CorleoneOED.SciMLBase.successful_retcode(result)
@assert isfinite(result.objective)
_, reconstruct = pack(default_vectorizer(), ps)
optimal_ps = reconstruct(result.u)
optimal_trajectory, _ = oed(nothing, optimal_ps, st)
optimal_F = fisher_information(oed, optimal_trajectory)
optimal_budgets = sampling_sums(oed, optimal_trajectory)
@assert isposdef(Symmetric(optimal_F))
@assert all(optimal_budgets .<= 4.0 + 1.0e-5)
@assert isapprox(result.objective, ACriterion()(optimal_F); rtol = 1.0e-5)
(; termination = result.retcode, objective = result.objective, budgets = optimal_budgets)

# Reconstruct from the current result vector and solve again: the old trajectory's
# saved states do not change when parameter containers change.
trajectoryplot(optimal_trajectory; skips = augmented_states, state_string = "Population")

# Optimized populations and schedules. Densities may be fractional: this is a
# continuous relaxation, not a binary sampling certificate or global-optimum proof.

# ## Carry information into another experiment
# Re-evaluate the completed experiment with its own parameter/state tree and add
# its information to the prior. Existing priors are not counted a second time.
next_st = update_fim(oed, [(; ps = optimal_ps, st)], st)
@assert next_st.F_init ≈ optimal_F
next_st.F_init
# Prepare the next design using `ps=optimal_ps, st=next_st`.
