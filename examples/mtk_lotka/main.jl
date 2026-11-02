#src ---
#src title: ModelingToolkit Integration
#src description: Define and solve an optimal control problem with ModelingToolkit
#src tags:
#src   - Lotka Volterra
#src   - Optimal Control
#src   - ModelingToolkit
#src icon: 🎣
#src ---

# # [ModelingToolkit Integration](@id mtk)
# This tutorial models the [Lotka–Volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem)
# with [ModelingToolkit.jl](https://docs.sciml.ai/ModelingToolkit/stable/).
# We solve it using single and multiple shooting, as in the
# [numerical fishing example](@ref lotka_fishing).

# ## [Setup](@id mtk_setup)
# ModelingToolkit defines the dynamics and cost. Corleone constructs the shooting
# layer, ComponentArrays supplies the optimizer's parameter vector, and ForwardDiff
# computes derivatives. We solve with Ipopt and plot with CairoMakie.
using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using ComponentArrays, ForwardDiff
using Optimization
using OptimizationMOI, Ipopt
using LuxCore, Random
using CairoMakie

# ## Model

# The states `x` and `y` are prey and predator populations. Their global initial
# values are fixed because `tunable = false`. The input `u` is fishing intensity,
# bounded between zero and one. Population bounds apply to lifted state values
# when using multiple shooting; they do not impose continuous path constraints.

@variables begin
    x(..) = 0.5, [tunable = false, bounds = (0.0, 5.0)]
    y(..) = 0.7, [tunable = false, bounds = (0.0, 5.0)]
    u(..) = 0.0, [bounds = (0.0, 1.0), input = true]
end
@constants begin
    c₁ = 0.4
    c₂ = 0.2
end

# Fishing removes both populations at different rates.
eqs = [
    D(x(t)) ~ x(t) - x(t) * y(t) - c₁ * u(t) * x(t)
    D(y(t)) ~ -y(t) + x(t) * y(t) - c₂ * u(t) * y(t)
]

# Minimize the accumulated squared deviations of both populations from one over
# the time horizon [0, 12]. Corleone introduces a quadrature state for this integral.
costs = [
    Symbolics.Integral(t in (0.0, 12.0))(
        (x(t) - 1.0)^2 + (y(t) - 1.0)^2
    ),
]
@named lotka_volterra = System(eqs, t, costs = costs)

# ## Single Shooting

# Supply the system, initial-value overrides, and the control schedule. The empty
# vector keeps the system defaults. The leading zero denotes the first control
# interval; this grid gives 120 piecewise constant values, one per 0.1 time units.
# The integral determines the time span, so it need not be specified separately.
dynopt = CorleoneDynamicOptProblem(
    lotka_volterra, [],
    u(t) => 0.0:0.1:11.9,
    algorithm = Tsit5(),
)

# Construct the optimization problem with the control bounds from the model.
optprob = OptimizationProblem(dynopt, AutoForwardDiff())

# Use ForwardDiff derivatives and Ipopt's limited-memory Hessian approximation.
uopt = solve(
    optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
    hessian_approximation = "limited-memory",
    print_level = 3
)
@assert successful_retcode(uopt)

# Restore the parameter structure, evaluate the optimized layer, and plot it.
_, reconstruct = pack(default_vectorizer(), dynopt.ps)
optimized_parameters = reconstruct(uopt.u)
optsol, _ = dynopt.layer(nothing, optimized_parameters, optprob.p)
trajectoryplot(optsol)

# ## Multiple Shooting

# Use 48 control intervals and lift the states at every interior interval boundary.
# Interior shooting values remain tunable even though the global initial values
# are fixed. Continuity equalities connect neighboring segments, including the
# quadrature state, so the objective still accounts for the whole time horizon.
# Population bounds now constrain the lifted values at every shooting point;
# overly restrictive bounds can make this formulation infeasible.
dynopt = CorleoneDynamicOptProblem(
    lotka_volterra, [],
    u(t) => 0.0:0.25:11.75,
    algorithm = Tsit5(),
    shooting = collect(0.25:0.25:11.75),
    problem_kwargs = (; abstol = 1.0e-9, reltol = 1.0e-9)
)

# Corleone appends the shooting equalities to the optimization constraints.
optprob = OptimizationProblem(dynopt, AutoForwardDiff())

optsol = solve(
    optprob,
    Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300,
    print_level = 3
)
@assert successful_retcode(optsol)

# Reconstruct and plot the multiple-shooting solution in the same way.
_, reconstruct_ms = pack(default_vectorizer(), dynopt.ps)
optimized_parameters = reconstruct_ms(optsol.u)
optsol, _ = dynopt.layer(nothing, optimized_parameters, optprob.p)
trajectoryplot(optsol)

# Check continuity after optimization. Small residuals connect the lifted physical
# states and quadrature; integration and optimizer tolerances are separate controls
# on the numerical result. The two formulations use different control grids, so
# their objective values are not an exact discretization-to-discretization check.
maximum(abs, shooting_constraints(optsol))
