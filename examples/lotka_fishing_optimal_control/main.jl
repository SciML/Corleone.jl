#src ---
#src title: The Lotka Volterra Fishing Problem
#src description: Optimize fishing intensity in a predator-prey model
#src tags:
#src   - Lotka Volterra
#src   - Optimal Control
#src icon: 🎣
#src ---

# ## [Lotka-Volterra Fishing Problem](@id lotka_fishing)
# This example is based on the [Lotka-Volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem).

# ## [Setup](@id lotka_oc_setup)

# Use Corleone to define the shooting layers and optimization models.
using Corleone

# The other packages provide:

# - Layer initialization: `LuxCore` and `Random`
# - ODE integration: `OrdinaryDiffEqTsit5`
# - Optimization: `Optimization`, `OptimizationMOI`, `Ipopt`, and `ComponentArrays`
# - Objective and constraint gradients: `ForwardDiff`
# - Plotting: `CairoMakie`

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
Makie.inline!()
# ## Lotka–Volterra Dynamics

# The first two states are prey and predator populations. The third accumulates
# their squared deviations from one. The first parameter is fishing intensity.

function lotka_dynamics(du, u, p, t)
    du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    return du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
end
tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [0.0, 1.0, 1.0]
prob = ODEProblem(lotka_dynamics, u0, tspan, p0; abstol = 1.0e-9, reltol = 1.0e-9)

# ## Single Shooting Approach

# Make fishing intensity tunable between zero and one, and keep the two
# interaction rates fixed. The 119 switching times define 120 control intervals.
# The initial state is taken from the ODE problem.
cgrid = collect(0.1:0.1:11.9)
control = Tunable(
    1; kind = Val(:parameter), default = 1.0,
    lowerbound = 0.0, upperbound = 1.0
) => cgrid
physical_parameters = (
    Constant(2; kind = Val(:parameter)) => nothing,
    Constant(3; kind = Val(:parameter)) => nothing,
)
layer = ShootingLayer(
    prob, Tsit5(), control, physical_parameters...;
    controls = [1], quadratures = [3]
)

# Minimize the terminal accumulated cost. The model initializes parameters and
# state, then prepares the objective. OptimizationProblem applies the control bounds
# from the annotation and uses ForwardDiff for gradients.
dynprob = CorleoneDynamicOptProblem(
    layer, :(u₃(12.0))
)
opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
uopt = solve(
    opt,
    Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300,
    print_level = 3
);

# Reconstruct the optimized parameters and plot the trajectory.
optimized_parameters = opt.f.f.reconstruct(uopt.u)
optsol, _ = layer(nothing, optimized_parameters, opt.p)
trajectoryplot(optsol)

# ## Multiple Shooting

# Lift all three states at interior shooting points. Matching constraints enforce
# continuity, including the accumulated cost, so its final value covers the whole
# time horizon. Full-state resets allow independent blocks with `EnsembleThreads()`.
# Start Julia with `--threads=4` to use four threads.

# Start with control intervals of length 0.25. The shooting points at 3.4 and 6.1
# lie inside intervals; 9.75 already coincides with a control switch.
shooting_points = [3.4, 6.1, 9.75]
initial_conditions = (
    Tunable(i; kind = Val(:variable), lowerbound = 0.0) => shooting_points for i in 1:3
)
mslayer = ShootingLayer(
    prob, Tsit5(), first(control) => collect(0.25:0.25:11.75), physical_parameters...,
    initial_conditions...; ensemblealg = EnsembleThreads(),
    controls = [1], quadratures = [3]
)

# Initialize controls to one and lifted states to the corresponding ODE initial values.
msps, msst = LuxCore.setup(MersenneTwister(42), mslayer)

# Split the control intervals at 3.4 and 6.1, duplicating their current values.
msps, msst = add_parameter_segments!(mslayer, msps, msst)

# Plot the initial segments and their reset jumps.
mssol, _ = mslayer(nothing, msps, msst);
trajectoryplot(mssol, show_legend = true, whiskerwidth = 10)

# Reuse the expanded parameters and state. State continuity and matching control
# values at inserted switches are included as equality constraints.
ms_dynprob = CorleoneDynamicOptProblem(
    mslayer, :(u₃(12.0)), ps = msps, st = msst
)

ms_opt = OptimizationProblem(ms_dynprob, AutoForwardDiff(); function_kwargs = (;))
ms_uopt = solve(
    ms_opt,
    Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300,
    print_level = 3
);

# Reconstruct the optimized parameters and plot the trajectory.
optimized_parameters = ms_opt.f.f.reconstruct(ms_uopt)
ms_optsol, _ = mslayer(nothing, optimized_parameters, ms_opt.p)
trajectoryplot(ms_optsol)
