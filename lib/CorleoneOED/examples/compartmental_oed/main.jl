#src ---
#src title: Discrete measurements for a compartmental model
#src description: Allocate measurement weights at candidate sampling times.
#src ---

# ## [Compartmental OED](@id compartmental_oed)
# Estimate three parameters by observing one state. We optimize a relaxed
# selection of six measurements from 19 candidate times, following the
# [compartmental OED benchmark](https://mintoc.de/index.php?title=Compartmental_OED).
using CorleoneOED, LuxCore, Random, OrdinaryDiffEqTsit5
using Optimization, OptimizationMOI, Ipopt, ComponentArrays, CairoMakie
using SymbolicIndexingInterface: parameter_values_at_time, parameter_index

function compartmental(du, u, p, t)
    θ1, θ2, θ3 = p
    du[1] = θ3 * (-θ1 * exp(-θ1 * t) + θ2 * exp(-θ2 * t))
    return nothing
end
problem = ODEProblem(compartmental, [0.0], (0.0, 50.0), [0.05884, 4.298, 21.8])
measurement_points = [
    0.0, 0.166, 0.333, 0.5, 0.666, 1.0, 1.5, 2.0, 2.5, 3.0,
    4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 24.0, 30.0, 48.0,
]
oed = OEDLayer(
    problem, Tsit5(); params = [1, 2, 3], measurements = (
        DiscreteMeasurement(:sample, measurement_points, (x, p, t) -> x[1]),
    )
)
ps, st = LuxCore.setup(Xoshiro(1), oed)

# ## Solve the relaxed design
# Physical parameters remain fixed. Each candidate time has a weight in `[0,1]`;
# the total weight is bounded by six. Fractional weights are allowed in this solve.
optprob = OptimizationProblem(oed, DCriterion(); ps, st, M = [6.0])
result = solve(
    optprob, Ipopt.Optimizer(); tol = 1.0e-8,
    hessian_approximation = "limited-memory", max_iter = 100, print_level = 0
)
_, reconstruct = pack(default_vectorizer(), ps)
trajectory, _ = oed(nothing, reconstruct(result.u), st)
F = fisher_information(oed, trajectory)
weights = [
    parameter_values_at_time(trajectory, trajectory, t)[parameter_index(trajectory, :sample)]
        for t in measurement_points
]
figure = Figure()
axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "Measurement weight")
scatter!(axis, measurement_points, weights)
figure

# ## Integer sampling decisions
# To require binary selections, construct the problem below and choose an
# integer-capable optimizer. Ipopt solves the continuous relaxation only.
integer_problem = OptimizationProblem(oed, DCriterion(); ps, st, M = [6.0], integer_weights = true)
