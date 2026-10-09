#src ---
#src title: Universal Differential Equation with a Lux Network
#src description: Embed a Lux.Chain of Dense layers in the right-hand side of an ODE to replace an unknown term, then train it with Adam followed by L-BFGS-B through the CorleoneParameterEstimation Optimization extension
#src tags:
#src   - Universal Differential Equation
#src   - Neural Network
#src   - Parameter Estimation
#src   - Lux
#src icon: 🧠
#src ---

# ## [Universal differential equations with Corleone](@id corleone_pe_ude)
#
# A **universal differential equation (UDE)** keeps the mechanistic part of a model that we
# trust and replaces the part we do not understand by a neural network. Writing the state
# `u` and the known vector field `f`, the model reads
# ```math
# \dot u = f(u, t) + \mathrm{NN}_\theta(u),
# ```
# where `\mathrm{NN}_\theta` is a small neural network whose parameters `\theta` are fitted
# jointly with the ODE. Because the network enters the *right-hand side*, it is trained
# through the ODE solver, not on a pre-computed derivative.
#
# In this example the unknown term is a scalar nonlinear reaction. The true dynamics are
# ```math
# \dot u = -0.5\,u + g(u), \qquad g(u) = 0.5\,\sin(u),
# ```
# and we only tell the model the linear decay `-0.5u`; the network has to discover `g` from
# noisy observations of `u(t)`.
#
# The example shows the two ingredients that make a UDE fit work in practice:
#
# 1. **Embedding a `Lux` network in the ODE parameters.** The network parameters are stored
#    inside the problem parameters `p` as a `ComponentArray` field, so Corleone's shooting
#    layer flattens and optimizes them like any other model parameter.
# 2. **A two-stage training schedule.** A short **Adam** phase finds a good basin of the
#    non-convex network landscape, and a quasi-Newton **L-BFGS-B** phase then polishes the
#    solution to high accuracy. Both stages are driven through the
#    `CorleoneParameterEstimation` Optimization extension.

# ## Setup

using Corleone, CorleoneParameterEstimation
using Lux, LuxCore
using ComponentArrays
using OrdinaryDiffEqTsit5
using Optimization, OptimizationOptimisers, OptimizationLBFGSB
using CairoMakie
using StableRNGs

# ## Ground truth and noisy observations

# We define the true right-hand side, integrate it on a fine grid, and add Gaussian
# measurement noise. The noise scale `σ` is stored in the data so that the fit uses
# *weighted* least-squares residuals `(y_obs - y_model) / σ`.

g_true(u) = 0.5 * sin(u)

function true_rhs!(du, u, p, t)
    du[1] = -0.5 * u[1] + g_true(u[1])
    return
end

u0 = [2.0]
tspan = (0.0, 10.0)
prob_true = ODEProblem(true_rhs!, u0, tspan, Float64[])
tdata = collect(0.0:0.2:10.0)

rng = StableRNG(1)
traj_true = solve(prob_true, Tsit5(); saveat = tdata)
ydata = reduce(hcat, traj_true.u) .+ 0.01 .* randn(rng, 1, length(tdata))
data = ExperimentData(tdata, ydata; σ = 0.01)

# ## The neural network
#
# We use a `Lux.Chain` of three `Dense` layers (20 and 8 hidden units) with a `tanh`
# activation. The output layer is **zero-initialized**, so at the start of training the
# network contributes nothing and the UDE reduces to the known linear decay. This is
# important: a randomly initialized network can produce a right-hand side large enough to
# make the ODE solver blow up before the first gradient step.
#
# Lux creates `Float32` parameters by default; we promote them to `Float64` to match the
# state and the rest of the model.

nn = Lux.Chain(Lux.Dense(1, 20, tanh), Lux.Dense(20, 8, tanh), Lux.Dense(8, 1))

ps_nn, st_nn = LuxCore.setup(rng, nn)
ps_nn = Lux.fmap(x -> Float64.(x), ps_nn)
ps_nn = (
    layer_1 = ps_nn.layer_1,
    layer_2 = ps_nn.layer_2,
    layer_3 = (weight = zero(ps_nn.layer_3.weight), bias = zero(ps_nn.layer_3.bias)),
)

# ## Embedding the network in the right-hand side
#
# The trick is to put the network parameters into the ODE parameter container `p` under a
# dedicated field `net`. A `ComponentArray` gives that container a flat vector view while
# keeping the nested `net.layer_1.weight`, ... access in the right-hand side. Corleone's
# `SingleShootingLayer` then treats *every* entry of this flat vector as a tunable model
# parameter, so the network weights are optimized together with the rest of the model.

p0 = ComponentArray(; net = ComponentArray(ps_nn))

function ude_rhs!(du, u, p, t)
    nn_out = nn([u[1]], p.net, st_nn)[1][1]
    du[1] = -0.5 * u[1] + nn_out
    return
end

prob = ODEProblem(ude_rhs!, u0, tspan, p0)

# ## The parameter-estimation layer
#
# `ParameterEstimationLayer` wraps the ODE problem and the data. The resulting
# `OptimizationProblem` exposes a flat vector of all tunable parameters — here the 217
# weights and biases of the network — together with the scalar objective
# ``\tfrac12 \sum_i r_i^2`` over the weighted residuals.

pe = ParameterEstimationLayer(prob, Tsit5(), data)
optprob = OptimizationProblem(pe)
println("Number of tunable parameters: ", length(optprob.u0))

# ## Two-stage training
#
# ### Stage 1 — Adam
#
# Adam is a robust first-order method for the rough, non-convex landscape of a randomly
# connected network. We run a fixed budget of iterations and keep the result as the starting
# point for the second stage.

sol_adam = solve(optprob, Adam(0.01); maxiters = 500)
println("Adam objective: ", sol_adam.objective)

# ### Stage 2 — L-BFGS-B
#
# Starting from the Adam iterate, a limited-memory BFGS solver converges quickly to a
# high-accuracy least-squares solution.

sol = solve(optprob, LBFGSB(); u0 = sol_adam.u, maxiters = 500)
println("L-BFGS-B objective: ", sol.objective)
println("Converged: ", SciMLBase.successful_retcode(sol))

# ## Reconstructing the fitted trajectory
#
# The optimization variables are exactly the flattened parameter vector (the model has no
# tunable initial condition), so we can rebuild the fitted `ComponentArray` from the
# solution and re-integrate the UDE. We then measure the weighted coefficient of
# determination ``R^2``.

p_fit = ComponentArray(sol.u, getaxes(p0))
traj_fit = solve(remake(prob; p = p_fit), Tsit5(); saveat = tdata)
yfit = reduce(hcat, traj_fit.u)

r2 = 1 - sum(abs2, ydata .- yfit) / sum(abs2, ydata .- sum(ydata) / length(ydata))
println("Weighted R² = ", r2)

# ## What did the network learn?
#
# The most informative check for a UDE is not the trajectory fit but the *learned term*
# itself. We evaluate the network on a grid of states and compare it against the true
# `g(u)`. A neural network is only constrained where it has seen data, so the shaded region
# in the right-hand plot marks the states visited by the trajectory; outside it the network
# is extrapolating and we should not trust its output.

ugrid = collect(0.0:0.05:2.0)
g_hat = [nn([u], p_fit.net, st_nn)[1][1] for u in ugrid]
u_obs = vec(ydata)
u_lo, u_hi = extrema(u_obs)
println("states visited by the data: [", round(u_lo; digits = 3), ", ",
    round(u_hi; digits = 3), "]")
println("max |ĝ(u) - g(u)| on the data range = ",
    maximum(abs.(g_hat[u_lo .<= ugrid .<= u_hi] .- g_true.(ugrid[u_lo .<= ugrid .<= u_hi]))))

# ## Plotting

function plot_ude(tdata, ydata, traj_true, yfit, ugrid, g_hat)
    f = Figure(size = (1100, 420))
    u_obs = vec(ydata)
    u_lo, u_hi = extrema(u_obs)

    ax_traj = CairoMakie.Axis(
        f[1, 1];
        xlabel = "time t",
        ylabel = "state u",
        title = "Trajectory fit (weighted R² = $(round(r2; digits = 4)))",
    )
    lines!(ax_traj, tdata, vec(reduce(hcat, traj_true.u)); linestyle = :dash,
        color = :black, label = "true")
    scatter!(ax_traj, tdata, vec(ydata); color = (:steelblue, 0.6), markersize = 6,
        label = "observations")
    lines!(ax_traj, tdata, vec(yfit); linewidth = 2.5, color = :crimson,
        label = "UDE fit")
    axislegend(ax_traj; position = :rt)

    ax_term = CairoMakie.Axis(
        f[1, 2];
        xlabel = "state u",
        ylabel = "unknown term",
        title = "Recovered term",
    )

    lines!(ax_term, ugrid, g_true.(ugrid); linestyle = :dash, color = :black,
        label = "g(u) = 0.5 sin(u)")
    lines!(ax_term, ugrid, g_hat; linewidth = 2.5, color = :crimson,
        label = "network ĝ(u)")
    scatter!(ax_term, u_obs, g_true.(u_obs); color = (:steelblue, 0.75), marker=:x, markersize = 10,
        label = "training data")
    extrapolation = false
    if u_lo > minimum(ugrid)
        extrapolation = true
        vspan!(ax_term, minimum(ugrid), u_lo; color = (:gray, 0.18), label = "extrapolation")
    end
    if u_hi < maximum(ugrid)
        vspan!(ax_term, u_hi, maximum(ugrid); color = (:gray, 0.18), label = extrapolation ? nothing : "extrapolation")
    end
    axislegend(ax_term; position = :rb)
    return f
end

plot_ude(tdata, ydata, traj_true, yfit, ugrid, g_hat)
