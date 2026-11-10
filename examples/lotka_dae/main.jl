#src ---
#src title: The Lotka Volterra Fishing Problem with an Index-1 DAE
#src description: Optimize fishing intensity in a predator-prey model written as an index-1 DAE
#src tags:
#src   - Lotka Volterra
#src   - Optimal Control
#src   - DAE
#src icon: 🎣
#src ---

# # [Lotka–Volterra Fishing Problem as a DAE](@id lotka_dae)
# This example rewrites the [Lotka-Volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem)
# as an index-1 DAE and is based on [`lotka_fishing_optimal_control`](@ref lotka_fishing).

# ## [Setup](@id lotka_dae_setup)

# Use Corleone to define the shooting layer and optimization models.
using Corleone

# The other packages provide:

# - Layer initialization: `LuxCore` and `Random`
# - DAE integration: `OrdinaryDiffEqRosenbrock`
# - Optimization: `Optimization`, `OptimizationMOI`, `Ipopt`, and `ComponentArrays`
# - Objective and constraint gradients: `ForwardDiff`
# - Plotting: `CairoMakie`

using LuxCore
using Random
using OrdinaryDiffEqRosenbrock
using SymbolicIndexingInterface
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using ForwardDiff
using LinearAlgebra
using CairoMakie
using SciMLBase
Makie.inline!()
# ## Index-1 DAE Dynamics

# The populations `x` and `y` are differential states. Their effective growth rates
# `z` and `w` are algebraic states, enforced by the mass matrix. The fishing control
# `v` enters the algebraic equations; the fifth state accumulates the squared
# deviations of the populations from one.

# ```math
# \begin{aligned}
# \dot{x} &= xz, & x(0)&=0.5, \\
# \dot{y} &= yw, & y(0)&=0.7, \\
# 0 &= z - (1 - y - 0.4v), \\
# 0 &= w - (-1 + x - 0.2v), \\
# \dot{J} &= (x-1)^2+(y-1)^2, & J(0)&=0, & 0&\le v\le1.
# \end{aligned}
# ```

const C1 = 0.4
const C2 = 0.2

function lotka_dae(du, u, p, t)
    x, y, z, w, cost = u
    v = p[1]
    du[1] = x * z
    du[2] = y * w
    du[3] = z - (1.0 - y - C1 * v)
    du[4] = w - (-1.0 + x - C2 * v)
    return du[5] = (x - 1.0)^2 + (y - 1.0)^2
end

# The mass matrix marks `z` and `w` as algebraic. The initial guesses of the
# algebraic states are consistent with the initial control guess `v = 0`.
tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.3, -0.5, 0.0]
p0 = [1.0]
# The lifted algebraic states are optimization variables, so intermediate
# optimizer iterates make the initial state inconsistent. Skip the DAE
# initialization check; the constraints enforce consistency at the optimum.
prob = ODEProblem(
    ODEFunction(lotka_dae; mass_matrix = diagm([1.0, 1.0, 0.0, 0.0, 1.0])),
    u0, tspan, p0; abstol = 1.0e-9, reltol = 1.0e-9,
    initializealg = SciMLBase.NoInit()
)

# ## Single Shooting Approach

# Make fishing intensity tunable between zero and one, and lift the algebraic
# states at the initial time: their consistent values depend on the first control
# value, so they are optimization variables tied to it by the algebraic equations
# at `t = 0`. The 119 switching times define 120 control intervals.
cgrid = collect(0.1:0.1:11.9)
control = Tunable(
    1; kind = Val(:parameter), default = 0.0,
    lowerbound = 0.0, upperbound = 1.0
) => cgrid
algebraic_initial_conditions = (
    Tunable(i; kind = Val(:variable)) => nothing for i in 3:4
)
layer = ShootingLayer(
    prob, Rodas5P(), control, algebraic_initial_conditions...;
    controls = [1], quadratures = [5]
)

# The initial control value is the first control interval's value; it is read from
# the trajectory's retained parameters. The constraint residual is zero at the
# initial guess and ties the lifted algebraic states to the control throughout.
dae_initialization = traj -> begin
    start = first(state_values(traj))
    v0 = traj.ps.parameter.parameters.p₁.parameter.ps[1]
    [
        start[3] - (1.0 - start[2] - C1 * v0),
        start[4] - (-1.0 + start[1] - C2 * v0),
    ]
end

# Minimize the terminal accumulated cost. The model initializes parameters and
# state, then prepares the objective and the DAE initialization constraints.
dynprob = CorleoneDynamicOptProblem(
    layer, :(u₅(12.0)), dae_initialization;
    lcons = [0.0, 0.0], ucons = [0.0, 0.0]
)
opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
uopt = solve(
    opt,
    Ipopt.Optimizer(),
    tol = 1.0e-8,
    hessian_approximation = "limited-memory",
    max_iter = 1000,
    print_level = 3
);
@assert successful_retcode(uopt)

# Reconstruct the optimized parameters and plot the trajectory.
_, reconstruct = pack(default_vectorizer(), dynprob.ps)
optimized_parameters = reconstruct(uopt.u)
optsol, _ = layer(nothing, optimized_parameters, opt.p)
trajectoryplot(optsol)

# The optimized cost of about 1.3443 matches the known optimum of the fishing
# problem on the same control grid.
println("objective: ", uopt.objective)
println("maximum DAE initialization residual: ", maximum(abs, dae_initialization(optsol)))
