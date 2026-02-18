#src ---
#src title: ModelingToolkit Integration
#src description: Using SciMLs symbolic stack
#src tags:
#src   - Lotka Volterra
#src   - Optimal Control
#src   - ModelingToolkit
#src icon: 🎣
#src ---

# ## [ModelingToolkit Integration](@id mtk)
# This is a quick intro based on [the lotka volterra fishing problem](https://mintoc.de/index.php?title=Lotka_Volterra_fishing_problem).
# It is similar to [the classical fishing example](@ref lotka_fishing), but we will use [ModelingToolkit.jl]() to model it.

# ## [Setup](@id mtk_setup)
# We will use `Corleone` to define our optimal control problem.
using Corleone
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEqTsit5
using ComponentArrays, ForwardDiff
using Optimization
using OptimizationMOI, Ipopt
using LuxCore, Random
using CairoMakie

# We start by defining our system

@variables begin
    x(..) = 0.5, [tunable = false]
    y(..) = 0.7, [tunable = false]
    u(..) = 0.0, [bounds = (0.0, 1.0), input = true]
end
@constants begin
    c₁ = 0.4
    c₂ = 0.2
end
eqs = [
    D(x(t)) ~ x(t) - x(t) * y(t) - c₁ * u(t) * x(t)
    D(y(t)) ~ -y(t) + x(t) * y(t) - c₂ * u(t) * y(t)
]
costs = [
    Symbolics.Integral(t in (0.0, 12.0))(
        (x(t) - 1.0)^2 + (y(t) - 1.0)^2
    ),
]
@named lotka_volterra = System(eqs, t, costs = costs)

function plot_lotka(sol)
    f = Figure()
    ax = CairoMakie.Axis(f[1, 1])
    scatterlines!(ax, sol, vars = [1, 2, 3])
    f[1, 2] = Legend(f, ax, "States", framevisible = false)
    ax1 = CairoMakie.Axis(f[2, 1])
    stairs!(ax1, sol, vars = [4,])
    f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
    return f
end

# ## Single Shooting

dynopt = CorleoneDynamicOptProblem(
    lotka_volterra, [],
    u(t) => 0.0:0.1:11.9,
    algorithm = Tsit5(),
)

optprob = OptimizationProblem(dynopt, AutoForwardDiff(), Val(:ComponentArrays))

sol = solve(
    optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
    hessian_approximation = "limited-memory"
)


u_opt = ComponentVector(sol.u, optprob.f.f.ax)
opt_traj, _ = dynopt.layer(nothing, u_opt, LuxCore.initialstates(Random.default_rng(), dynopt.layer))

plot_lotka(opt_traj)
# ## Multiple Shooting
dynopt = CorleoneDynamicOptProblem(
    lotka_volterra, [],
    u(t) => 0.0:0.1:11.9,
    algorithm = Tsit5(),
    shooting = [0.0, 3.0, 6.0, 9.0]
)

optprob = OptimizationProblem(dynopt, AutoForwardDiff(), Val(:ComponentArrays))

sol = solve(
    optprob, Ipopt.Optimizer(), max_iter = 1000, tol = 5.0e-6,
    hessian_approximation = "limited-memory"
)

u_opt = ComponentVector(sol.u, optprob.f.f.ax)
opt_traj, _ = dynopt.layer(nothing, u_opt, LuxCore.initialstates(Random.default_rng(), dynopt.layer))

plot_lotka(opt_traj)
