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
# This example rewrites the
# [Lotka–Volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem)
# as an index-1 DAE. It builds on the [fishing tutorial](@ref lotka_fishing) and shows
# how to handle algebraic states whose consistent values depend on a control.

# ## [Setup](@id lotka_dae_setup)

# Corleone defines the shooting layer and the optimization model. The other
# packages provide layer initialization (`LuxCore`, `Random`), DAE integration
# (`OrdinaryDiffEqRosenbrock`), optimization (`Optimization`, `OptimizationMOI`,
# `Ipopt`, `ComponentArrays`), derivatives (`ForwardDiff`) and plots (`CairoMakie`).
using Corleone

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
#
# ```math
# \begin{aligned}
# \dot{x} &= xz, & x(0)&=0.5, \\
# \dot{y} &= yw, & y(0)&=0.7, \\
# 0 &= z - (1 - y - c_1 v), \\
# 0 &= w - (-1 + x - c_2 v), \\
# \dot{J} &= (x-1)^2+(y-1)^2, & J(0)&=0, & 0&\le v\le1.
# \end{aligned}
# ```
#
# The parameter vector is `p = [v, c₁, c₂]` with the catch coefficients
# `c₁ = 0.4` and `c₂ = 0.2`.

function lotka_dae(du, u, p, t)
    x, y, z, w, cost = u
    v, c₁, c₂ = p
    du[1] = x * z
    du[2] = y * w
    du[3] = z - (1.0 - y - c₁ * v)
    du[4] = w - (-1.0 + x - c₂ * v)
    du[5] = (x - 1.0)^2 + (y - 1.0)^2
    return nothing
end

# The mass matrix marks `z` and `w` as algebraic. The initial guesses of the
# algebraic states are consistent with the initial control guess `v = 0`.
tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.3, -0.5, 0.0]
p0 = [0.0, 0.4, 0.2]

# !!! note "Skipping the DAE initialization"
#     The lifted algebraic states are optimization variables, so intermediate
#     optimizer iterates make the initial state inconsistent. We therefore skip
#     the solver's DAE initialization; explicit constraints enforce consistency
#     at the optimum instead.
prob = ODEProblem(
    ODEFunction(lotka_dae; mass_matrix = diagm([1.0, 1.0, 0.0, 0.0, 1.0])),
    u0, tspan, p0; abstol = 1.0e-9, reltol = 1.0e-9,
    initializealg = SciMLBase.NoInit()
)

# ## Single Shooting Approach

# Make the fishing intensity tunable between zero and one on 120 intervals, and
# keep the catch coefficients `Constant`. The consistent values of the algebraic
# states depend on the first control value, so we lift them at the initial time:
# they become optimization variables tied to the control by the algebraic
# equations at `t = 0`.
cgrid = collect(0.1:0.1:11.9)
control = Tunable(
    1; kind = Val(:parameter), default = 0.0,
    lowerbound = 0.0, upperbound = 1.0
) => cgrid
catch_coefficients = Constant([2, 3]; kind = Val(:parameter)) => nothing
algebraic_initial_conditions = (
    Tunable(i; kind = Val(:variable)) => nothing for i in 3:4
)
layer = ShootingLayer(
    prob, Rodas5P(), control, catch_coefficients, algebraic_initial_conditions...;
    controls = [1], quadratures = [5]
)

# The two initialization constraints are the algebraic equations at `t = 0`,
# written as symbolic expressions of states (`uᵢ`) and parameters (`pᵢ`). Their
# residual is zero at the initial guess.
dae_initialization = (
    :(u₃(0.0) - (1.0 - u₂(0.0) - p₂(0.0) * p₁(0.0))),
    :(u₄(0.0) - (-1.0 + u₁(0.0) - p₃(0.0) * p₁(0.0))),
)

# Minimize the terminal accumulated cost subject to the initialization equalities.
dynprob = CorleoneDynamicOptProblem(
    layer, :(u₅(12.0)), dae_initialization...;
    lcons = [0.0, 0.0], ucons = [0.0, 0.0]
)
opt = OptimizationProblem(dynprob, AutoForwardDiff(); function_kwargs = (;))
uopt = solve(
    opt, Ipopt.Optimizer();
    tol = 1.0e-8, hessian_approximation = "limited-memory",
    max_iter = 1000, print_level = 0
)
@assert successful_retcode(uopt)

# Reconstruct the optimized parameters and plot the trajectory.
_, reconstruct = pack(default_vectorizer(), dynprob.ps)
optimized_parameters = reconstruct(uopt.u)
optsol, _ = layer(nothing, optimized_parameters, opt.p)
trajectoryplot(optsol)

# The optimized cost of about 1.3443 matches the known optimum of the fishing
# problem on the same control grid, and the algebraic equations hold at `t = 0`.
(;
    retcode = uopt.retcode,
    objective = uopt.objective,
    initialization_residual = maximum(
        abs, optimization_constraints(
            dynprob, optimized_parameters, opt.p
        )
    ),
)
