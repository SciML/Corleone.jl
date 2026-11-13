# ---
# title: Multiple-shooting parameter estimation
# description: Fit fishing-model rates and a shooting-node state with exact continuity constraints.
# category: parameter_estimation
# id: inverse_multiple_shooting
# julia: "1.10"
# notebook: false
# generate_cover: false
# ---

# Multiple shooting splits a trajectory into segments that are integrated
# independently. This makes fits more robust for oscillatory or unstable models.
# Here we partition the fishing-model trajectory at ``t = 6`` and estimate the node
# state together with the two interaction rates. The applied OED input is known.
#
# !!! info "Prerequisites"
#     This example reuses the design and data of the
#     [OED-controlled example](@ref inverse_lotka_oed_control).
using Corleone: SciMLBase
using CorleoneInverse, LuxCore, ComponentArrays, OrdinaryDiffEqTsit5, NonlinearSolve
using Optimization, OptimizationMOI, Ipopt, StableRNGs, CairoMakie, LinearAlgebra
include(joinpath(pkgdir(CorleoneInverse), "examples", "lotka_oed_control", "design.jl"))

rng = StableRNG(7)
design = fishing_design(rng)
data = fishing_observations(design, design.controls, rng)

# ## A single-shooting fit as a starting point
# The rates `β, δ` are tunable within moderate bounds; the control schedule and the
# catch coefficients are `Constant`.
parameters = (
    Tunable(2:3; default = [1.3, 0.8], lowerbound = 0.1, upperbound = 3.0) => nothing,
    Constant(1; default = design.controls) => design.switches,
    catch_coefficients,
)
single = ShootingLayer(design.problem, Tsit5(), parameters...; controls = [1])
ps_1, st_1 = LuxCore.setup(rng, single)
single_model = CorleoneDynamicOptProblem(single, data; ps = ps_1, st = st_1)
fit_1 = solve(NonlinearLeastSquaresProblem(single_model), LevenbergMarquardt())
@assert SciMLBase.successful_retcode(fit_1)
_, reconstruct_1 = pack(default_vectorizer(), ps_1)
initial_trajectory, _ = single(
    data, reconstruct_1(fit_1.u), experiment_state(single, data, st_1)
)
node = initial_trajectory.u[findfirst(==(6.0), initial_trajectory.t)]
(; single_shooting_rates = ParameterEstimate(single_model, fit_1.u).x, node)

# ## Build the constrained problem
# A variable annotation at the interior grid point introduces the node unknowns,
# initialized from the single-shooting trajectory. Finite bounds keep the rate and
# node trial values in a moderate range. `OptimizationProblem` adds continuity at
# the node as equality constraints.
multiple = ShootingLayer(
    design.problem, Tsit5(), parameters...,
    Tunable(
        1:2; kind = Val(:variable), default = node, lowerbound = 0.0, upperbound = 10.0
    ) => [6.0];
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

# !!! tip "Penalties vs. constraints"
#     The original PR #168 example penalized the shooting gaps in the residual.
#     Exact equality constraints keep the fitted trajectory continuous without a
#     penalty weight to tune.

# The continuity residual at the node should vanish:
_, reconstruct = pack(default_vectorizer(), ps)
trajectory, _ = multiple(data, reconstruct(fit.u), experiment_state(multiple, data, st))
gap = shooting_constraints(trajectory)
@assert norm(gap) < 1.0e-6
gap

figure = Figure()
axis = CairoMakie.Axis(figure[1, 1]; xlabel = "Time", ylabel = "Population")
for (i, observation) in enumerate(data.observations)
    lines!(axis, trajectory.t, getindex.(trajectory.u, i); label = "Fitted species $i")
    scatter!(axis, observation.t, observation.y; label = "Observed species $i")
end
vlines!(axis, [6.0]; color = :gray, linestyle = :dash, label = "Shooting node")
axislegend(axis)
figure

# ## Acknowledgements
#
# This example is adapted from Christoph Plate
# ([@chplate](https://github.com/chplate))'s example in
# [Corleone.jl PR #168](https://github.com/SciML/Corleone.jl/pull/168).
