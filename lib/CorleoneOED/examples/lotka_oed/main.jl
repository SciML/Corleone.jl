#src ---
#src title: The Lotka Volterra Optimal Experimental Design Problem
#src description: A classic example to optimal experimental design for ODEs
#src tags:
#src   - Lotka Volterra
#src   - Optimal Experimental Design
#src icon: 🎣
#src ---

# This is a quick intro based on [the Lotka Volterra Experimental Design Problem](https://mintoc.de/index.php?title=Lotka_Experimental_Design).

# ## Setup
# We will use `Corleone` and `CorleoneOED`to model the optimal experimental design problem.
using Corleone, CorleoneOED

# Additionally, we will need the folllowing packages
# - [`LuxCore`]() and [`Random`]() for basic setup functions
# - [`OrdinaryDiffEqTsit5`]() as an adaptive solver for the related ODEProblem
# - [`SymbolicIndexingInterface`]() to conviniently access variables and controls of the solution
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
using CairoMakie

# ## Lotka Volterrra Dynamics

function lotka_dynamics(du, u, p, t)
    du[1] = u[1] - p[2] * prod(u[1:2]) - 0.4 * p[1] * u[1]
    du[2] = -u[2] + p[3] * prod(u[1:2]) - 0.2 * p[1] * u[2]
    return
end
tspan = (0.0, 12.0)
u0 = [0.5, 0.7]
p0 = [0.0, 1.0, 1.0]
prob = ODEProblem(lotka_dynamics, u0, tspan, p0)

# ## Construct the `OEDLayer`

# For optimal experimental design problems, `CorleoneOED` provides the `OEDLayer`. Like the
# `SingleShootingLayer` and the `MultipleShootingLayer`, the `OEDLayer` is a callable layer,
# that integrates the problem with the specified integrator and applies the piecewise
# constant controls to it. The `OEDLayer` however also adds differential states and
# differential equations for a) the forward sentitivities of the solution with respect to
# the parameters and b) the Fisher information matrix to the dynamical system.
# To construct it, we again define first the piecewise constant control discretization on
# a control grid.

cgrid = collect(0.0:0.1:11.9)
control = ControlParameter(
    cgrid, name=:fishing, bounds=(0.0, 1.0), controls=ones(length(cgrid))
)

# Now, there are different possibilities to construct the `OEDLayer`. It can be constructed
# either directly from the `ODEProblem`, or we can first build a `SingleShootingLayer`
# from the problem and with that the `OEDLayer`.

oed = OEDLayer{false}(
    prob, Tsit5(),
    params = [2, 3],
    controls = (1 => control,),
    bounds_p=([1.0, 1.0], [1.0, 1.0]),
    measurements = [
        ControlParameter(collect(0.0:0.25:11.75), controls = 0.5 * ones(48), bounds = (0.0, 1.0)),
        ControlParameter(collect(0.0:0.25:11.75), controls = 0.5 * ones(48), bounds = (0.0, 1.0)),
    ],
    observed = (u, p, t) -> u[1:2],
)

layer = Corleone.SingleShootingLayer(prob, Tsit5(), controls=(1 => control,), bounds_p=([1.0, 1.0], [1.0, 1.0]))

oed = OEDLayer{false}(
    layer,
    params = [2, 3],
    measurements = [
        ControlParameter(collect(0.0:0.25:11.75), controls = 0.5 * ones(48), bounds = (0.0, 1.0)),
        ControlParameter(collect(0.0:0.25:11.75), controls = 0.5 * ones(48), bounds = (0.0, 1.0)),
    ],
    observed = (u, p, t) -> u[1:2],
)
# ## Set up and solve the problem

# With the `OEDLayer` set up, we can define the `OptimizationProblem` in one line by
# giving a suitable criterion to minimize, e.g., the `DCriterion`. Here, the determinant of the
# inverse of the Fisher information matrix is minimized. An upper bound `M` on the maximum
# time of measurements is specified via `M`.

optprob = OptimizationProblem(oed, ACriterion(); M=[4.0, 4.0])
uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol=1.0e-6,
    hessian_approximation="limited-memory",
    max_iter=3
);

# After solving, we now only need to investigate the solution.
ps, st = LuxCore.setup(Random.default_rng(), oed)
optsol, _ = oed(nothing, uopt + zero(ComponentArray(ps)), st)

function plot_lotka(sol)
    f = Figure()
    ax = CairoMakie.Axis(f[1, 1], xticks = 0:2:12, title = "States")
    ax2 = CairoMakie.Axis(f[1, 2], xticks = 0:2:12, title = "Sensitivities")
    ax3 = CairoMakie.Axis(f[2, 1], xticks = 0:1:12, title = "Sampling")
    ax4 = CairoMakie.Axis(f[2, 2], xticks = 0:1:12, title = "Controls")
    plot!(ax, sol, idxs = [1, 2])
    plot!(ax2, sol, idxs = [3, 4, 5, 6])
    stairs!(ax3, sol, vars=[:w₁, :w₂])
    stairs!(ax4, sol, vars = [:p₁])
    f
end

plot_lotka(optsol)
