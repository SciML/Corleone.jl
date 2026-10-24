#src ---
#src title: The Lotka Volterra Fishing Problem
#src description: A classic example to relaxed mixed-integer optimal control
#src tags:
#src   - Lotka Volterra
#src   - Optimal Control
#src icon: 🎣
#src ---

# ## [Lotka-Volterra Fishing Problem](@id lotka_fishing)
# This is a quick intro based on [the lotka volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem).

# ## [Setup](@id lotka_oc_setup)
# We will use `Corleone` to define our optimal control problem.
using Corleone

# Additionally, we will need the following packages
# - [`LuxCore`]() and [`Random`]() for basic setup functions
# - [`OrdinaryDiffEqTsit5`]() as an adaptive solver for the related ODEProblem
# - [`SymbolicIndexingInterface`]() to conveniently access variables and controls of the solution
# - [`Optimization`](), [`OptimizationMOI`](), [`Ipopt`](), and [`ComponentArrays`]() to setup and solve the optimization problem
# - [`CairoMakie`]() to plot the solution

using LuxCore
using Random
using OrdinaryDiffEqTsit5
using SymbolicIndexingInterface
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using ForwardDiff
using CairoMakie

# ## Lotka Volterrra Dynamics

# First, in the ODEProblem has to be defined in a standard way.

function lotka_dynamics(du, u, p, t)
    du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    return du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
end
tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [0.0, 1.0, 1.0]
prob = ODEProblem(lotka_dynamics, u0, tspan, p0; abstol = 1e-9, reltol = 1e-9)

# ## Single Shooting Approach

# Define fishing intensity as a tunable parameter and the two physical rates as
# constants. The grid contains interior switching times; there is one more
# control value than grid entries. Initial states pass through unchanged.
cgrid = collect(0.1:0.1:11.9)
control = Tunable(1; kind = Val(:parameter), default = 1.0,
    lowerbound = 0.0, upperbound = 1.0) => cgrid
physical_parameters = (
    Constant(2; kind = Val(:parameter)) => nothing,
    Constant(3; kind = Val(:parameter)) => nothing,
)
layer = ShootingLayer(prob, Tsit5(), control, physical_parameters...)
ps, st = LuxCore.setup(MersenneTwister(42), layer)
initial_parameters = ComponentArray(ps)
parameter_axes = getaxes(initial_parameters)

function plot_lotka(sol)
    f = Figure()
    ax = CairoMakie.Axis(f[1, 1])
    lines!(ax, sol.t, sol[:u₁], label = "prey")
    lines!(ax, sol.t, sol[:u₂], label = "predator")
    f[1, 2] = Legend(f, ax, "States", framevisible = false)
    ax1 = CairoMakie.Axis(f[2, 1])
    controls = [SymbolicIndexingInterface.parameter_values_at_time(sol, sol, t)[1] for t in sol.t]
    stairs!(ax1, sol.t, controls, label = "fishing", step = :post)
    f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
    return f
end

# The third state integrates the running cost. Construct the objective and bounds
# explicitly: annotations describe the definitions, while Optimization.jl receives
# the flattened numerical bounds. Only fishing controls are tunable here.
function fishing_objective(q, state)
    parameters = ComponentArray(q, parameter_axes)
    trajectory, _ = layer(nothing, parameters, state)
    return last(trajectory.u)[3]
end
optfun = OptimizationFunction(fishing_objective, AutoForwardDiff())
optprob = OptimizationProblem(optfun, initial_parameters, st;
    lb = zeros(length(initial_parameters)), ub = ones(length(initial_parameters)))
uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300,
)
optimized_parameters = ComponentArray(uopt.u, parameter_axes)
optsol, _ = layer(nothing, optimized_parameters, st)

plot_lotka(optsol)

# ## Multiple Shooting

# We can also partition the time horizon into smaller intervals on which the integration
# is decoupled. At these shooting points, variables are introduced for the initial values
# of the differential states. By this lifting approach, nonlinearities can be reduced.
# Use the same `ShootingLayer`, adding per-variable initial conditions at the
# interior shooting points. Every variable, including accumulated cost, is lifted.
# Setting their matching residuals to zero enforces continuity of all three states,
# so the final cost still represents the integral over the entire horizon.
# Full-state resets also permit independent blocks to run with `EnsembleThreads()`.
# Start Julia with `--threads=4` (or your desired count) to use several threads.

shooting_points = [3.0, 6.0, 9.0]
initial_conditions = (
    Tunable(i; kind = Val(:variable)) => shooting_points for i in 1:3
)
mslayer = ShootingLayer(prob, Tsit5(), control, physical_parameters...,
    initial_conditions...; ensemblealg = EnsembleThreads())
msps, msst = LuxCore.setup(MersenneTwister(42), mslayer)

# Seed the lifted values from a continuous forward trajectory. Here we reuse the
# single-shooting result, including its optimized control values.
msps.parameter.parameters.p₁.parameter.ps .= optimized_parameters.parameter.parameters.p₁.parameter.ps
for (i, ic) in enumerate(values(msps.initialcondition.initialconditions))
    ic.parameter.ps .= [optsol.u[findfirst(==(t), optsol.t)][i] for t in shooting_points]
end
ms_initial_parameters = ComponentArray(msps)
ms_parameter_axes = getaxes(ms_initial_parameters)

function ms_objective(q, state)
    trajectory, _ = mslayer(nothing, ComponentArray(q, ms_parameter_axes), state)
    return last(trajectory.u)[3]
end
function continuity_constraints!(res, q, state)
    trajectory, _ = mslayer(nothing, ComponentArray(q, ms_parameter_axes), state)
    res .= reduce(vcat, trajectory.residuals)
    return nothing
end

# Bound lifted population/cost values below by zero; only fishing has an upper
# bound of one. The first initial state stays fixed because there is no lift at t=0.
ms_lower = zero(ms_initial_parameters)
ms_upper = similar(ms_initial_parameters)
fill!(ms_upper, Inf)
ms_upper.parameter.parameters.p₁.parameter.ps .= 1.0
nconstraints = length(u0) * length(shooting_points)
ms_optfun = OptimizationFunction(ms_objective, AutoForwardDiff(); cons = continuity_constraints!)
ms_optprob = OptimizationProblem(ms_optfun, ms_initial_parameters, msst;
    lb = ms_lower, ub = ms_upper,
    lcons = zeros(nconstraints), ucons = zeros(nconstraints))
ms_uopt = solve(
    ms_optprob, Ipopt.Optimizer(),
    tol = 1.0e-5,
    hessian_approximation = "limited-memory",
    max_iter = 300,
)
mssol, _ = mslayer(nothing, ComponentArray(ms_uopt.u, ms_parameter_axes), msst)
plot_lotka(mssol)
