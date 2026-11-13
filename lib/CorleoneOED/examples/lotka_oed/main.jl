# ---
# title: Lotka–Volterra experimental design
# description: Jointly optimize fishing and continuous measurement schedules.
# id: lotka_oed
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# Let's design an experiment that estimates the two interaction coefficients of
# a Lotka–Volterra model as precisely as possible. Fishing is a physical control,
# and two measurement densities select when to observe each population. This
# follows the
# [Lotka experimental-design benchmark](https://mintoc.de/index.php?title=Lotka_Experimental_Design).
#
# !!! info "Prerequisites"
#     Constructing a Corleone shooting layer and the
#     [measurement and information conventions](@ref oed_formulation).

using CorleoneOED, LuxCore, Random, OrdinaryDiffEqTsit5, LinearAlgebra
using Optimization, OptimizationMOI, Ipopt, ComponentArrays, CairoMakie
using SymbolicIndexingInterface: variable_symbols, variable_index

# The parameter vector is `p = [v, β, δ, c₁, c₂]`: the fishing control, the two
# interaction coefficients, and the fixed catch coefficients `c₁ = 0.4` and
# `c₂ = 0.2`.
function lotka_dynamics(du, x, p, t)
    v, β, δ, c₁, c₂ = p
    du[1] = x[1] - β * x[1] * x[2] - c₁ * v * x[1]
    du[2] = -x[2] + δ * x[1] * x[2] - c₂ * v * x[2]
    return nothing
end
problem = ODEProblem(lotka_dynamics, [0.5, 0.7], (0.0, 12.0), [0.0, 1.0, 1.0, 0.4, 0.2])

# We estimate `β` and `δ` (indices `[2, 3]`) at their nominal values; only the
# fishing control and the measurement weights are design variables.
#
# !!! note "Estimated parameters vs. design variables"
#     `params` selects the parameters whose sensitivities determine the
#     information. They stay fixed at their nominal values during the design. The
#     [formulation](@ref oed_formulation) explains this distinction.
#
# No physical shooting resets are introduced in this tutorial, so the automatic
# shooting-equality block is empty.
#
# ## Define controls and measurements
# The fishing control has one value per interval, including the first one. The
# catch coefficients are `Constant`. Unannotated parameters (here `β` and `δ`)
# also remain fixed at their nominal values.
control = Tunable(
    1; default = 0.0, lowerbound = 0.0, upperbound = 1.0
) => collect(0.25:0.25:11.75)
catch_coefficients = Constant([4, 5]) => nothing
measurements = (
    ContinuousMeasurement(:w1, collect(0.0:0.25:11.75), (x, p, t) -> x[1]),
    ContinuousMeasurement(:w2, collect(0.0:0.25:11.75), (x, p, t) -> x[2]),
)
oed = OEDLayer(
    problem, Tsit5(), control, catch_coefficients;
    params = [2, 3], measurements, controls = [1]
)
ps, st = LuxCore.setup(Xoshiro(1), oed);
trajectory, _ = oed(nothing, ps, st);
# Keep the physical states visible instead of overlaying the augmented G/F states.
augmented_states = filter(variable_symbols(trajectory)) do symbol
    variable_index(trajectory, symbol) > length(problem.u0)
end
trajectoryplot(trajectory; skips = augmented_states, state_string = "Population")

# The plot shows the initial populations, fishing, and unit measurement densities.
# Controls and measurement weights are parameter histories, not appended physical
# states: the legend's `p₁` is fishing, while `w1` and `w2` are the two densities.

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
# Minimize the trace of the inverse information (the A-criterion) with at most
# four time units of measurement for each population. The facade adds the
# sampling constraints after user constraints and before automatic shooting
# equalities. All grids stay fixed during optimization.
#
# !!! tip "Optional packages"
#     ComponentArrays supplies the shared vectorizer; Optimization, OptimizationMOI,
#     and Ipopt the continuous optimizer; and CairoMakie the plots. None of them is
#     required just to simulate an OED layer.
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

# These are the optimized populations and schedules. Nice: fishing now excites
# the dynamics, and the densities concentrate where the populations are most
# informative.
#
# !!! note "Relaxed designs"
#     Densities may be fractional. This is a continuous relaxation, not a binary
#     sampling certificate or a proof of global optimality.

# ## Carry information into another experiment
# Re-evaluate the completed experiment with its own parameter/state tree and add
# its information to the prior. Existing priors are not counted a second time.
next_st = update_fim(oed, [(; ps = optimal_ps, st)], st)
@assert next_st.F_init ≈ optimal_F
next_st.F_init

# Prepare the next design using `ps = optimal_ps, st = next_st`.
