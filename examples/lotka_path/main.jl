#src ---
#src title: The Lotka Volterra Fishing Problem with a Terminal Inequality
#src description: Optimize fishing intensity with a penalized terminal biomass constraint and a path constraint on the predator
#src tags:
#src   - Lotka Volterra
#src   - Optimal Control
#src   - Path Constraints
#src icon: 🎣
#src ---

# # [Lotka–Volterra Fishing Problem with a Terminal Inequality](@id lotka_path)
# This example adds a terminal inequality constraint to the
# [Lotka–Volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem).
# A terminal prey biomass below ``1.1`` is allowed but penalized through the slack
# variable ``s`` in the objective. On top of the mintoc formulation we add a path
# constraint that keeps the predator above ``0.1`` on the saved time grid. The
# control is relaxed to ``w \in [0, 1]``.
#
# !!! info "What you will learn"
#     How to combine a scalar tunable parameter (the slack), a path constraint over
#     a whole state series, and a terminal constraint in one problem.

# ## [Setup](@id lotka_path_setup)

using Corleone

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

# ## Dynamics

# The first two states are prey and predator biomasses. The third state
# accumulates the squared deviations of the biomasses from one, which turns the
# integral objective into a terminal value. The parameter vector is
# `p = [w, s, c₁, c₂]`: the fishing control, the slack, and the fixed catch
# coefficients `c₁ = 0.4` and `c₂ = 0.2`.
#
# ```math
# \begin{aligned}
# \dot{x}_0 &= x_0 - x_0x_1 - c_1x_0w, & x_0(0)&=0.5, \\
# \dot{x}_1 &= -x_1 + x_0x_1 - c_2x_1w, & x_1(0)&=0.7, \\
# \dot{x}_2 &= (x_0-1)^2+(x_1-1)^2, & x_2(0)&=0, \\
# x_0(12) &\ge 1.1 - s, & s&\ge0, & 0&\le w\le1.
# \end{aligned}
# ```

function lotka_dynamics(du, u, p, t)
    w, s, c₁, c₂ = p
    du[1] = u[1] - u[1] * u[2] - c₁ * w * u[1]
    du[2] = -u[2] + u[1] * u[2] - c₂ * w * u[2]
    du[3] = (u[1] - 1.0)^2 + (u[2] - 1.0)^2
    return nothing
end

tspan = (0.0, 12.0)
u0 = [0.5, 0.7, 0.0]
p0 = [0.0, 0.0, 0.4, 0.2]
prob = ODEProblem(lotka_dynamics, u0, tspan, p0; abstol = 1.0e-9, reltol = 1.0e-9)

# ## Single Shooting Approach

# The fishing control is piecewise constant on 120 intervals, the slack `s` is a
# free scalar bounded below by zero, and the catch coefficients are `Constant`.
cgrid = collect(0.1:0.1:11.9)
control = Tunable(
    1; kind = Val(:parameter), default = 0.5,
    lowerbound = 0.0, upperbound = 1.0
) => cgrid
slack = Tunable(
    2; kind = Val(:parameter), default = 0.0,
    lowerbound = 0.0, upperbound = Inf
) => nothing
catch_coefficients = Constant([3, 4]; kind = Val(:parameter)) => nothing
layer = ShootingLayer(
    prob, Tsit5(), control, slack, catch_coefficients;
    controls = [1], quadratures = [3]
)

# The objective is the terminal accumulated cost plus the slack penalty. The path
# constraint keeps the predator above ``0.1``; the terminal constraint is the
# penalized biomass inequality.
#
# !!! tip "Series vs. point evaluations"
#     A bare signal such as `u₂` denotes the full series on the saved time grid,
#     so `:(u₂ .- 0.1)` yields one constraint per saved time. A call such as
#     `u₁(12.0)` evaluates a single point.
npoints = length(cgrid) + 2
dynprob = CorleoneDynamicOptProblem(
    layer, :(u₃(12.0) + 10 * p₂(0.0)),
    :(u₂ .- 0.1),
    :(u₁(12.0) - 1.1 + p₂(0.0));
    lcons = [zeros(npoints), [0.0]], ucons = [fill(Inf, npoints), [Inf]]
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

# The optimized cost approaches the mintoc reference value of about 1.3655 for
# the relaxed problem on a fine grid. The slack stays at zero whenever the
# terminal biomass constraint can be satisfied for free.
(;
    objective = uopt.objective,
    slack = max(first(optsol[:p₂]), 0.0),
    terminal_prey = last(optsol.u)[1],
    minimum_predator = minimum(getindex.(optsol.u, 2)),
)
