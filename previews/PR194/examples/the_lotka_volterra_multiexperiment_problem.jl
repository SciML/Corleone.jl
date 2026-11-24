using CorleoneOED, LuxCore, Random, OrdinaryDiffEqTsit5, LinearAlgebra
using Optimization, OptimizationMOI, Ipopt, ComponentArrays, CairoMakie
using SymbolicIndexingInterface: variable_symbols, variable_index

function lotka_dynamics(x, p, t)
    v, β, δ, c₁, c₂ = p
    return [
        x[1] - β * x[1] * x[2] - c₁ * v * x[1],
        -x[2] + δ * x[1] * x[2] - c₂ * v * x[2],
    ]
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

multi = MultiExperimentLayer(oed, 2)
ps, st = LuxCore.setup(Xoshiro(1), multi);

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

augmented_states = filter(variable_symbols(trajectories.experiment_1)) do symbol
    variable_index(trajectories.experiment_1, symbol) > length(problem.u0)
end
trajectoryplot(
    trajectories.experiment_1; skips = augmented_states, state_string = "Population"
)

trajectoryplot(
    trajectories.experiment_2; skips = augmented_states, state_string = "Population"
)

prior_state = merge(st, (; F_init = Matrix{Float64}(I, 2, 2)))
F_with_prior, _ = fisher_information(multi, nothing, optimal_ps, prior_state)
@assert F_with_prior ≈ F + prior_state.F_init
next_st = update_fim(multi, [(; ps = optimal_ps, st)], prior_state)
@assert next_st.F_init ≈ prior_state.F_init + F
next_st.F_init

first_design = OEDLayer(
    problem, Tsit5(), control, catch_coefficients; params = [2], measurements, controls = [1]
)
second_design = OEDLayer(
    problem, Tsit5(), control, catch_coefficients; params = [3], measurements, controls = [1]
)
split_design = MultiExperimentLayer((first_design, second_design))
split_ps, split_st = LuxCore.setup(Xoshiro(2), split_design)
split_problem = OptimizationProblem(
    split_design, ACriterion();
    ps = split_ps, st = split_st, M = fill(4.0, 4)
);

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
