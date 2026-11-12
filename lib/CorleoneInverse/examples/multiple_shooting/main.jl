# ---
# title: Multiple-shooting parameter estimation with fixed controls
# description: Fit fishing-model rates and a shooting-node state with exact continuity constraints.
# category: parameter_estimation
# id: inverse_multiple_shooting
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# Adapted from Christoph Plate (@chplate)'s example in
# [Corleone.jl PR #168](https://github.com/SciML/Corleone.jl/pull/168).

# Partition the fishing-model trajectory at t = 6 and estimate the node state
# together with its two interaction rates. The applied OED input is known.
using Corleone: SciMLBase
using CorleoneInverse, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5, NonlinearSolve
using Optimization, OptimizationMOI, Ipopt, StableRNGs, CairoMakie, LinearAlgebra
include(joinpath(pkgdir(CorleoneInverse), "examples", "lotka_oed_control", "design.jl"))

rng = StableRNG(7)
design = fishing_design(rng)
data = fishing_observations(design, design.controls, rng)
parameters = (
    Tunable(2:3; default = [1.3, 0.8], lowerbound = 0.1, upperbound = 3.0) => nothing,
    Constant(1; default = design.controls) => design.switches,
)
single = ShootingLayer(design.problem, Tsit5(), parameters...; controls = [1])
ps_1, st_1 = LuxCore.setup(rng, single)
single_model = CorleoneDynamicOptProblem(single, data; ps = ps_1, st = st_1)
fit_1 = solve(NonlinearLeastSquaresProblem(single_model), LevenbergMarquardt())
@assert SciMLBase.successful_retcode(fit_1)
_, reconstruct_1 = pack(default_vectorizer(), ps_1)
initial_trajectory, _ = single(data, reconstruct_1(fit_1.u), experiment_state(single, data, st_1))
node = initial_trajectory.u[findfirst(==(6.0), initial_trajectory.t)]

# ## Build the constrained problem
# A variable annotation at the interior grid point introduces the node unknowns.
# Finite natural bounds keep the rate and node trial values in a moderate range.
# OptimizationProblem adds continuity as equality constraints. This replaces the
# shooting-residual penalty used by the original PR #168 example.
multiple = ShootingLayer(
    design.problem, Tsit5(), parameters...,
    Tunable(1:2; kind = Val(:variable), default = node, lowerbound = 0.0, upperbound = 10.0) => [6.0];
    controls = [1]
)
ps, st = LuxCore.setup(rng, multiple)
model = CorleoneDynamicOptProblem(multiple, data; ps, st)
opt = OptimizationProblem(model, Optimization.AutoForwardDiff())
@assert length(opt.lcons) == 2
fit = solve(
    opt, Ipopt.Optimizer(); print_level = 0, tol = 1.0e-9,
    hessian_approximation = "limited-memory", max_iter = 500
)
@assert SciMLBase.successful_retcode(fit)
estimate = ParameterEstimate(model, fit.u)
println(estimate)
_, reconstruct = pack(default_vectorizer(), ps)
trajectory, _ = multiple(data, reconstruct(fit.u), experiment_state(multiple, data, st))
gap = shooting_constraints(trajectory)
println("Continuity residual = ", gap)
@assert norm(gap) < 1.0e-6

figure = Figure()
axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "Population")
for (i, observation) in enumerate(data.observations)
    lines!(axis, trajectory.t, getindex.(trajectory.u, i); label = "Fitted species $i")
    scatter!(axis, observation.t, observation.y; label = "Observed species $i")
end
vlines!(axis, [6.0]; color = :gray, linestyle = :dash, label = "Shooting node")
axislegend(axis)
figure
