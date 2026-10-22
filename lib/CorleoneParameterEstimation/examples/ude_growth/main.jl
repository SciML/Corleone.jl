#src ---
#src title: Universal Differential Equation with a Lux Network
#src description: Embed a Lux.Chain of Dense layers in the right-hand side of an ODE to replace an unknown term. Fit one experiment, see the extrapolation, then add a second experiment and fit both jointly with Adam followed by L-BFGS-B through the CorleoneParameterEstimation Optimization extension
#src tags:
#src   - Universal Differential Equation
#src   - Neural Network
#src   - Parameter Estimation
#src   - Multi-experiment
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
# where ``\mathrm{NN}_\theta`` is a small neural network whose parameters ``\theta`` are fitted
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
# We proceed in two steps. First we fit a **single experiment** and plot the recovered term:
# the network is only constrained where the trajectory has visited, and it extrapolates
# everywhere else. Then we add a **second experiment** from a different initial condition
# that visits the unobserved region and fit both experiments **jointly**, which pins the
# network down on the previously extrapolated part.

# ## Setup

using Corleone, CorleoneParameterEstimation
using Lux, LuxCore
using ComponentArrays
using OrdinaryDiffEqTsit5
using Optimization, OptimizationOptimisers, OptimizationLBFGSB
using LinearAlgebra: norm
using Statistics: mean
using CairoMakie
using StableRNGs

# ## Ground truth and the first experiment

# We define the true right-hand side and integrate it from `u₀ = 2.0`. The trajectory decays
# through the upper half of state space. Observations are corrupted with Gaussian
# measurement noise `σ = 0.01`, stored in the data so that the fit uses *weighted*
# least-squares residuals `(y_obs - y_model) / σ`.

g_true(u) = 0.5 * sin(u)

function true_rhs!(du, u, p, t)
    du[1] = -0.5 * u[1] + g_true(u[1])
    return
end

u0_1 = [2.0]
tspan = (0.0, 10.0)
tdata = collect(0.0:0.2:10.0)
σ_obs = 0.01

rng = StableRNG(1)
traj_true1 = solve(ODEProblem(true_rhs!, u0_1, tspan, Float64[]), Tsit5(); saveat = tdata)
y1 = reduce(hcat, traj_true1.u) .+ σ_obs .* randn(rng, 1, length(tdata))
data_1 = ExperimentData(tdata, y1; σ = σ_obs)

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

prob = ODEProblem(ude_rhs!, u0_1, tspan, p0)

# ## Fitting the first experiment
#
# `ParameterEstimationLayer` wraps the ODE problem and the data. The resulting
# `OptimizationProblem` exposes a flat vector of all tunable parameters — here the 217
# weights and biases of the network — together with the scalar objective
# ``\tfrac12 \sum_i r_i^2`` over the weighted residuals.
#
# Training uses two stages: a short **Adam** phase to find a good basin of the non-convex
# network landscape, followed by a quasi-Newton **L-BFGS-B** phase that polishes the
# solution. Both are driven through the `CorleoneParameterEstimation` Optimization
# extension.

pe_1 = ParameterEstimationLayer(prob, Tsit5(), data_1)
opt_1 = OptimizationProblem(pe_1)
println("Number of tunable parameters: ", length(opt_1.u0))

sol_adam_1 = solve(opt_1, Adam(0.01); maxiters = 500)
println("Adam objective (experiment 1): ", sol_adam_1.objective)

sol_1 = solve(opt_1, LBFGSB(); u0 = sol_adam_1.u, maxiters = 500)
println("L-BFGS-B objective (experiment 1): ", sol_1.objective)

# The optimization variables are exactly the flattened parameter vector (the model has no
# tunable initial condition), so we can rebuild the fitted `ComponentArray` from the
# solution.

p_1 = ComponentArray(sol_1.u, getaxes(p0))

# ## The first fit extrapolates
#
# We simulate the fitted model and evaluate its learned term on a grid of states. The
# shaded region marks the states the trajectory never visits: there the network is
# extrapolating. The fit is excellent on the visited range but the recovered term departs
# from `g(u)` below `u ≈ 0.74`.

function ude_traj(p; u0 = u0_1)
    return solve(remake(prob; p = p, u0 = u0), Tsit5(); saveat = tdata)
end

single_1 = reduce(hcat, ude_traj(p_1).u)

ugrid = collect(0.0:0.05:2.0)
g_single = [nn([u], p_1.net, st_nn)[1][1] for u in ugrid]

u_obs_1 = vec(y1)
u_lo_1, u_hi_1 = extrema(u_obs_1)
println(
    "States visited by experiment 1: [", round(u_lo_1; digits = 3), ", ",
    round(u_hi_1; digits = 3), "]"
)

function plot_experiment_1(tdata, y1, traj_true1, yfit1, ugrid, g_fit)
    f = Figure(size = (1100, 420))

    ax_traj = CairoMakie.Axis(
        f[1, 1];
        xlabel = "time t",
        ylabel = "state u",
        title = "Experiment 1 (u₀ = 2.0)",
    )
    lines!(
        ax_traj, tdata, vec(reduce(hcat, traj_true1.u)); linestyle = :dash,
        color = :black, label = "true"
    )
    scatter!(
        ax_traj, tdata, vec(y1); color = (:steelblue, 0.5), markersize = 5,
        label = "observations"
    )
    lines!(
        ax_traj, tdata, vec(yfit1); linewidth = 2.5, color = :crimson,
        label = "experiment-1 fit"
    )
    axislegend(ax_traj; position = :rt)

    ax_term = CairoMakie.Axis(
        f[1, 2];
        xlabel = "state u",
        ylabel = "unknown term",
        title = "Recovered reaction term",
    )
    u_obs = vec(y1)
    u_lo, u_hi = extrema(u_obs)

    lines!(
        ax_term, ugrid, g_true.(ugrid); linestyle = :dash, color = :black,
        label = "g(u) = 0.5 sin(u)"
    )
    lines!(
        ax_term, ugrid, g_fit; linewidth = 2.5, color = :crimson,
        label = "experiment-1 fit ĝ(u)"
    )
    scatter!(
        ax_term, u_obs, g_true.(u_obs); color = (:steelblue, 0.45), markersize = 8, marker = :x,
        label = "visited states"
    )
    extrapolate = false
    if u_lo > minimum(ugrid)
        extrapolate = true
        vspan!(ax_term, minimum(ugrid), u_lo; color = (:gray, 0.18), label = "extrapolation")
    end
    if u_hi < maximum(ugrid)
        vspan!(ax_term, u_hi, maximum(ugrid); color = (:gray, 0.18), label = extrapolate ? nothing : "extrapolation")
    end
    axislegend(ax_term; position = :rb)

    return f
end

plot_experiment_1(tdata, y1, traj_true1, single_1, ugrid, g_single)

# ## A second experiment
#
# To constrain the network where it was extrapolating, we run a second experiment from
# `u₀ = 0.3`, which stays in the low-state region experiment 1 never visited. The two
# experiments share the *same* network parameters but have their own initial conditions.
# Stacking the two `ExperimentData` objects builds the joint dataset.

u0_2 = [0.3]

traj_true2 = solve(ODEProblem(true_rhs!, u0_2, tspan, Float64[]), Tsit5(); saveat = tdata)
y2 = reduce(hcat, traj_true2.u) .+ σ_obs .* randn(rng, 1, length(tdata))

data_2 = ExperimentData(tdata, y2; σ = σ_obs, u0 = u0_2)
data_joint = ExperimentData(data_1, data_2)

println("Number of experiments: ", n_experiments(data_joint))
println("Total number of observations: ", n_observations(data_joint))

# ## Joint fit over both experiments
#
# The joint layer concatenates the residuals of both experiments and optimizes the *shared*
# network parameters. We warm-start from the first-experiment solution, so the second
# Adam/L-BFGS-B pass only has to correct the part of the network that experiment 1 could
# not identify.

pe_joint = ParameterEstimationLayer(prob, Tsit5(), data_joint)
opt_joint = OptimizationProblem(pe_joint)

sol_adam_joint = solve(opt_joint, Adam(0.01); u0 = sol_1.u, maxiters = 500)
println("Adam objective (joint): ", sol_adam_joint.objective)

sol_joint = solve(opt_joint, LBFGSB(); u0 = sol_adam_joint.u, maxiters = 500)
println("L-BFGS-B objective (joint): ", sol_joint.objective)

p_joint = ComponentArray(sol_joint.u, getaxes(p0))

# ## Evaluating both fits
#
# We simulate both experiments under each fit. The first-experiment fit drifts away on the
# second experiment, while the joint fit tracks both.

single_2 = reduce(hcat, ude_traj(p_1; u0 = u0_2).u)
joint_1 = reduce(hcat, ude_traj(p_joint).u)
joint_2 = reduce(hcat, ude_traj(p_joint; u0 = u0_2).u)

weighted_r2(yobs, ypred) = 1 - sum(abs2, yobs .- ypred) /
    sum(abs2, yobs .- sum(yobs) / length(yobs))
nrmse(yobs, ypred) = sqrt(sum(abs2, (yobs .- ypred) ./ σ_obs) / length(yobs))

println(
    "Weighted R² (experiment 1): experiment-1 fit = ", weighted_r2(y1, single_1),
    ", joint fit = ", weighted_r2(y1, joint_1)
)
println(
    "RMSE/σ (experiment 2):      experiment-1 fit = ", nrmse(y2, single_2),
    ", joint fit = ", nrmse(y2, joint_2)
)

# ## What did the joint fit learn?
#
# The joint fit recovers the term on the region visited by the second experiment, which the
# first-experiment fit had to extrapolate. Only the states below `u ≈ 0.29` remain
# unobserved and are still shaded.

g_joint = [nn([u], p_joint.net, st_nn)[1][1] for u in ugrid]

u_obs_all = vcat(vec(y1), vec(y2))
u_lo, u_hi = extrema(u_obs_all)
println(
    "States visited by the joint data: [", round(u_lo; digits = 3), ", ",
    round(u_hi; digits = 3), "]"
)
println(
    "Max |ĝ(u) - g(u)| on the joint data range: experiment-1 fit = ",
    maximum(abs.(g_single[u_lo .<= ugrid .<= u_hi] .- g_true.(ugrid[u_lo .<= ugrid .<= u_hi]))),
    ", joint fit = ",
    maximum(abs.(g_joint[u_lo .<= ugrid .<= u_hi] .- g_true.(ugrid[u_lo .<= ugrid .<= u_hi])))
)

# The left panel shows both experiments together: the experiment-1 fit (crimson) matches
# experiment 1 but leaves the second experiment, while the joint fit (dark green) follows
# both. The right panel compares the recovered terms.

function plot_joint_fit(
        tdata, y1, y2, traj_true1, traj_true2, joint_1, joint_2,
        ugrid, g_single, g_joint
    )
    f = Figure(size = (1200, 440))

    ax_traj = CairoMakie.Axis(
        f[1, 1];
        xlabel = "time t",
        ylabel = "state u",
        title = "Both experiments",
    )
    lines!(
        ax_traj, tdata, vec(reduce(hcat, traj_true1.u)); linestyle = :dash,
        color = :black, label = "true"
    )
    scatter!(
        ax_traj, tdata, vec(y1); color = (:steelblue, 0.5), markersize = 8, marker = :x,
        label = "observations (exp 1)"
    )
    lines!(
        ax_traj, tdata, vec(joint_1); linewidth = 2, color = :darkgreen,
        label = "joint fit"
    )
    lines!(
        ax_traj, tdata, vec(reduce(hcat, traj_true2.u)); linestyle = :dash,
        color = :black
    )
    scatter!(
        ax_traj, tdata, vec(y2); color = (:orange, 0.5), markersize = 8, marker = :x,
        label = "observations (exp 2)"
    )
    lines!(ax_traj, tdata, vec(joint_2); linewidth = 2, color = :darkgreen)
    axislegend(ax_traj; position = :rt)

    ax_term = CairoMakie.Axis(
        f[1, 2];
        xlabel = "state u",
        ylabel = "unknown term",
        title = "Recovered reaction term",
    )
    u_obs = vcat(vec(y1), vec(y2))
    u_lo, u_hi = extrema(u_obs)

    lines!(
        ax_term, ugrid, g_true.(ugrid); linestyle = :dash, color = :black,
        label = "g(u) = 0.5 sin(u)"
    )
    lines!(
        ax_term, ugrid, g_single; linewidth = 2, color = :crimson,
        label = "experiment-1 fit ĝ(u)"
    )
    lines!(
        ax_term, ugrid, g_joint; linewidth = 2, color = :darkgreen,
        label = "joint fit ĝ(u)"
    )
    scatter!(
        ax_term, vec(y1), g_true.(vec(y1)); color = (:steelblue, 0.35), markersize = 8, marker = :x,
        label = "visited (exp 1)"
    )
    scatter!(
        ax_term, vec(y2), g_true.(vec(y2)); color = (:orange, 0.35), markersize = 8, marker = :x,
        label = "visited (exp 2)"
    )
    extrapolate = false
    if u_lo > minimum(ugrid)
        extrapolate = true
        vspan!(ax_term, minimum(ugrid), u_lo; color = (:gray, 0.18), label = "extrapolation")
    end
    if u_hi < maximum(ugrid)
        vspan!(ax_term, u_hi, maximum(ugrid); color = (:gray, 0.18), label = extrapolate ? nothing : "extrapolation")
    end
    axislegend(ax_term; position = :rb)

    return f
end

plot_joint_fit(
    tdata, y1, y2, traj_true1, traj_true2, joint_1, joint_2,
    ugrid, g_single, g_joint,
)

# ## Regularizing the network with an L1 penalty
#
# The joint fit has **217 tunable network weights** but only
# ``n_{experiments}`` × `length(tdata)` scalar observations (here 2 × 51 = 102). With this many
# more unknowns than data points the parameters are only weakly identifiable: many weights
# can vary freely without changing the trajectory. This is the textbook setting for
# *regularization*.
#
# Corleone's [`L1Regularizer`](@ref) (lasso) augments the least-squares objective with a
# data-independent penalty ``\lambda\sum_i w_i |\theta_i|`` on the fitted parameters.
# Because the penalty appears as *additional residuals* appended to the observation
# residuals, it is completely transparent to the optimizer: the same Adam → L-BFGS-B scheme
# drives the penalized objective.
#
# By default only the **model parameters** are penalized — here all 217 weights — and any
# tuned initial conditions would be left untouched (this example has none). We simply pass
# the regularizer to `CorleoneParameterEstimationProblem` through its `regularization`
# keyword. The `regularization_reference` (shrinkage toward zero by default) and the
# number of residual rows appended are resolved automatically.

lasso_prob = CorleoneParameterEstimationProblem(pe_joint; regularization = L1Regularizer(1.0e-4))
println("Regularization residual rows added: ", lasso_prob.n_regularization_residuals)

# We warm-start from the unregularized joint fit and run the same two-stage optimization
# with the L1 penalty active.

lasso_opt = OptimizationProblem(pe_joint; regularization = L1Regularizer(1.0e-4))
sol_lasso = solve(lasso_opt, Adam(0.01); u0 = sol_joint.u, maxiters = 300)
sol_lasso = solve(lasso_opt, LBFGSB(); u0 = sol_lasso.u, maxiters = 500)
p_lasso = ComponentArray(sol_lasso.u, getaxes(p0))

# ## The L1-regularized fit
#
# With a mild penalty the trajectory fit is unchanged (the data dominates the objective),
# exactly as one would want from a small amount of regularization. The regularized fit is
# therefore our baseline fit for the rest of this example.

single_1_l1 = reduce(hcat, ude_traj(p_lasso).u)
single_2_l1 = reduce(hcat, ude_traj(p_lasso; u0 = u0_2).u)

println("L1-regularized weight vector norm: ", round(norm(Vector(p_lasso.net)); digits = 3))
println(
    "Weighted R² (experiment 1): joint fit = ",
    weighted_r2(y1, single_1_l1)
)
println(
    "RMSE/σ (experiment 2):      joint fit = ",
    nrmse(y2, vec(single_2_l1))
)

# The learned reaction term stays on top of the truth in the visited region.

g_lasso = [nn([u], p_lasso.net, st_nn)[1][1] for u in ugrid]

u_obs_all = vcat(vec(y1), vec(y2))
u_lo, u_hi = extrema(u_obs_all)
println(
    "Max |ĝ(u) - g(u)| on the joint data range (L1 fit): ",
    maximum(abs.(g_lasso[u_lo .<= ugrid .<= u_hi] .- g_true.(ugrid[u_lo .<= ugrid .<= u_hi])))
)

# The final panel compares the true term, the unregularized joint fit and the L1-regularized
# fit. The regularized fit is the baseline: it is visually indistinguishable from the
# unregularized one on the visited states, while the network parameters carry a small L1
# penalty.

function plot_regularized(tdata, ugrid, g_lasso)
    f = Figure(size = (900, 440))

    ax_traj = CairoMakie.Axis(
        f[1, 1];
        xlabel = "time t",
        ylabel = "state u",
        title = "L1-regularized fit",
    )
    scatter!(
        ax_traj, tdata, vec(y1); color = (:steelblue, 0.5), markersize = 8, marker = :x,
        label = "observations (exp 1)"
    )
    lines!(
        ax_traj, tdata, vec(single_1_l1); linewidth = 2.5, color = :darkgreen,
        label = "L1-regularized fit"
    )
    lines!(
        ax_traj, tdata, vec(reduce(hcat, traj_true1.u)); linestyle = :dash,
        color = :black, label = "true"
    )
    axislegend(ax_traj; position = :rt)

    ax_term = CairoMakie.Axis(
        f[1, 2];
        xlabel = "state u",
        ylabel = "unknown term",
        title = "Recovered reaction term",
    )
    lines!(
        ax_term, ugrid, g_true.(ugrid); linestyle = :dash, color = :black,
        label = "g(u) = 0.5 sin(u)"
    )
    lines!(
        ax_term, ugrid, g_lasso; linewidth = 2.5, color = :purple,
        label = "L1-regularized ĝ(u)"
    )
    scatter!(
        ax_term, vec(y2), g_true.(vec(y2)); color = (:orange, 0.35), markersize = 8, marker = :x,
        label = "visited (exp 2)"
    )
    axislegend(ax_term; position = :rb)

    return f
end

plot_regularized(tdata, ugrid, g_lasso)
