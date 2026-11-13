using CorleoneOED, LuxCore, Random, OrdinaryDiffEqTsit5, LinearAlgebra
using Optimization, OptimizationMOI, Ipopt, ComponentArrays, CairoMakie
using SymbolicIndexingInterface: variable_symbols, variable_index

function lotka_dynamics(du, x, p, t)
    v, β, δ, c₁, c₂ = p
    du[1] = x[1] - β * x[1] * x[2] - c₁ * v * x[1]
    du[2] = -x[2] + δ * x[1] * x[2] - c₂ * v * x[2]
    return nothing
end
problem = ODEProblem(lotka_dynamics, [0.5, 0.7], (0.0, 12.0), [0.0, 1.0, 1.0, 0.4, 0.2])

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

augmented_states = filter(variable_symbols(trajectory)) do symbol
    variable_index(trajectory, symbol) > length(problem.u0)
end
trajectoryplot(trajectory; skips = augmented_states, state_string = "Population")

F = fisher_information(oed, trajectory)
budgets = sampling_sums(oed, trajectory)
(; initial_information = F, initial_budgets = budgets)

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

trajectoryplot(optimal_trajectory; skips = augmented_states, state_string = "Population")

next_st = update_fim(oed, [(; ps = optimal_ps, st)], st)
@assert next_st.F_init ≈ optimal_F
next_st.F_init

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
