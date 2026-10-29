#src ---
#src title: Lotka–Volterra optimal experimental design
#src description: Jointly optimize fishing and continuous measurement schedules.
#src ---

# ## [Lotka–Volterra OED](@id lotka_oed)
# We design an experiment to estimate the two interaction coefficients of a
# Lotka–Volterra model. Fishing is a physical control; two measurement densities
# select when to observe each population. This follows the
# [Lotka experimental-design benchmark](https://mintoc.de/index.php?title=Lotka_Experimental_Design).

using CorleoneOED, LuxCore, Random, OrdinaryDiffEqTsit5
using Optimization, OptimizationMOI, Ipopt, ComponentArrays, CairoMakie

function lotka_dynamics(du, x, p, t)
    du[1] = x[1] - p[2] * x[1] * x[2] - 0.4 * p[1] * x[1]
    du[2] = -x[2] + p[3] * x[1] * x[2] - 0.2 * p[1] * x[2]
    return nothing
end
problem = ODEProblem(lotka_dynamics, [0.5, 0.7], (0.0, 12.0), [0.0, 1.0, 1.0])

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
ps, st = LuxCore.setup(Xoshiro(1), oed)
trajectory, _ = oed(nothing, ps, st)
trajectoryplot(trajectory)

# The layer integrates state sensitivities `G` and continuous information `F`:
# ```math
# G' = f_xG + f_\theta, \qquad
# F' = \sum_m w_m(t)J_m(t)^\top J_m(t), \qquad J_m=h_{m,x}G.
# ```
# Each weight lies in `[0,1]`. Its time integral is the measurement budget.
F = fisher_information(oed, trajectory)
budgets = sampling_sums(oed, trajectory)

# ## Optimize the design
# Minimize the trace of inverse information with at most four time units of
# measurement for each population. The facade adds the sampling constraints.
model = CorleoneDynamicOptProblem(oed, ACriterion(); ps, st, M = [4.0, 4.0])
optprob = OptimizationProblem(model, AutoForwardDiff())
result = solve(
    optprob, Ipopt.Optimizer(); tol = 1.0e-6,
    hessian_approximation = "limited-memory", max_iter = 100, print_level = 0
)
_, reconstruct = pack(default_vectorizer(), ps)
optimal_ps = reconstruct(result.u)
optimal_trajectory, _ = oed(nothing, optimal_ps, st)
trajectoryplot(optimal_trajectory)

# ## Carry information into another experiment
# Re-evaluate the completed experiment with its own parameter/state tree and add
# its information to the prior. Existing priors are not counted a second time.
next_st = update_fim(oed, [(; ps = optimal_ps, st)], st)
next_st.F_init
# Prepare the next design using `ps=optimal_ps, st=next_st`.
