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

# Additionally, we will need the following packages:
# - `LuxCore` and `Random` for layer initialization
# - `OrdinaryDiffEqTsit5` to solve the differential equation
# - `SymbolicIndexingInterface` to access trajectory variables and controls
# - `Optimization`, `OptimizationMOI`, `Ipopt`, and `ComponentArrays` to set up and solve the optimization problem
# - `ForwardDiff` to differentiate the objective and constraints
# - `CairoMakie` to plot the solution

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

# First, the ODEProblem has to be defined in a standard way.

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

# Define fishing intensity as a tunable parameter and the two physical rates as
# constants. The grid contains interior switching times; there is one more
# control value than grid entries. Initial states pass through unchanged.
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
ps, st = LuxCore.setup(MersenneTwister(42), layer)
initial_parameters = ComponentArray(ps)
parameter_axes = getaxes(initial_parameters)

# The third state integrates the running cost. The callbacks capture the concrete
# layer and parameter axes. The objective's scalar type assertion gives the
# optimizer an inferable return type, including ForwardDiff Dual numbers.
# The constraint callback writes state and parameter matching residuals directly
# into the optimizer's buffer.
function shooting_callbacks(layer, parameter_axes)
    function objective(q, state)
        trajectory, _ = layer(nothing, ComponentArray(q, parameter_axes), state)
        return last(trajectory.u)[3]::eltype(q)
    end
    function constraints!(res, q, state)
        trajectory, _ = layer(nothing, ComponentArray(q, parameter_axes), state)
        shooting_constraints!(res, trajectory)
        return nothing
    end
    return objective, constraints!
end

# Bounds follow the annotations and the initialized Lux parameter structure.
# Vectorize them with the same layout used for the optimizer parameters.
lower, upper = get_bounds(layer, ps, st)
fishing_objective, _ = shooting_callbacks(layer, parameter_axes)
optfun = OptimizationFunction(fishing_objective, AutoForwardDiff())
optprob = OptimizationProblem(
    optfun, initial_parameters, st;
    lb = ComponentArray(lower), ub = ComponentArray(upper)
)
uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300,
    print_level = 3,
)
optimized_parameters = ComponentArray(uopt.u, parameter_axes)
optsol, _ = layer(nothing, optimized_parameters, st)

trajectoryplot(optsol)

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

# Use a coarser control grid with quarter-unit intervals for multiple shooting.
# The first two shooting points split control intervals; the last already coincides
# with a control switch. This demonstrates both inserted and existing switches.
shooting_points = [3.4, 6.1, 9.75]
initial_conditions = (
    Tunable(i; kind = Val(:variable), lowerbound = 0.0) => shooting_points for i in 1:3
)
mslayer = ShootingLayer(
    prob, Tsit5(), first(control) => collect(0.25:0.25:11.75), physical_parameters...,
    initial_conditions...; ensemblealg = EnsembleThreads(),
    controls = [1], quadratures = [3]
)

# Setup initializes controls to one and each lifted state to the corresponding
# initial value of the problem.
msps, msst = LuxCore.setup(MersenneTwister(42), mslayer)

# Split control intervals at the first two shooting points. Each new interval
# duplicates its predecessor's value; the existing switch at 9.75 stays unchanged.
msps, msst = add_parameter_segments!(mslayer, msps, msst)

# Let's have a look at the initial solution
mssol, _ = mslayer(nothing, msps, msst);
trajectoryplot(mssol, show_legend = true, whiskerwidth = 10)

# Build optimizer storage after splitting, so its axes include the inserted values.
ms_initial_parameters = ComponentArray(msps)
ms_parameter_axes = getaxes(ms_initial_parameters)
ms_objective, continuity_constraints! = shooting_callbacks(mslayer, ms_parameter_axes)

# Annotation bounds cover lifted states and control values, including inserted
# copies. The count includes nine state and two parameter matching residuals.
# Constraint callbacks compute parameter differences on demand from the trajectory.
ms_lower, ms_upper = get_bounds(mslayer, msps, msst)
nconstraints = get_number_of_shooting_constraints(mslayer, msps, msst)
ms_optfun = OptimizationFunction(ms_objective, AutoForwardDiff(); cons = continuity_constraints!)
ms_optprob = OptimizationProblem(
    ms_optfun, ms_initial_parameters, msst;
    lb = ComponentArray(ms_lower), ub = ComponentArray(ms_upper),
    lcons = zeros(nconstraints), ucons = zeros(nconstraints)
)
ms_uopt = solve(
    ms_optprob, Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300,
    print_level = 3,
)
mssolop, _ = mslayer(nothing, ComponentArray(ms_uopt.u, ms_parameter_axes), msst);
trajectoryplot(mssolop, show_legend = true)
