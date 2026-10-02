#src ---
#src title: Joint Parameter Estimation over Multiple Experiments
#src description: Fit the Lotka-Volterra model parameters jointly from two experiments run under different initial conditions, showing how the parameter uncertainties shrink when more data is used
#src tags:
#src   - Lotka Volterra
#src   - Parameter Estimation
#src   - Multi-experiment
#src   - Least Squares
#src icon: 🎯
#src ---

# ## [Joint parameter estimation over multiple experiments](@id corleone_pe_lotka_multiexp)
#
# Parameter estimation is often performed iteratively: a first fit is done on an initial
# experiment, and later *new* data becomes available -- for example a second experiment run
# under *different initial conditions*. We then want to estimate the parameters from the
# *joint* dataset so that all available information is used to constrain the *shared* model
# parameters. Here we do exactly that with the Lotka-Volterra model: two experiments starting
# from different initial conditions, fitted together, and we show how the uncertainties on
# the shared parameters shrink relative to a fit on the first experiment alone.

# ## Setup

using Corleone, CorleoneParameterEstimation
using CairoMakie
using OrdinaryDiffEqTsit5
using NonlinearSolve
using StableRNGs

# ## Lotka-Volterra dynamics

# The classical two-species predator-prey model
# ```math
# \begin{aligned}
# \frac{dx}{dt} &= \alpha x - \beta x y,\\
# \frac{dy}{dt} &= -\delta y + \gamma x y,
# \end{aligned}
# ```
# with four parameters `p = (α, β, δ, γ)`.

function lotka_volterra!(du, u, p, t)
    x, y = u
    α, β, δ, γ = p
    du[1] = α * x - β * x * y
    du[2] = -δ * y + γ * x * y
    return
end

tspan = (0.0, 10.0)
p_true = [1.5, 1.0, 3.0, 1.0]
rng = StableRNG(42)
tsteps = collect(0.0:0.1:10.0)

# ### 1. First experiment
# Simulate a first experiment started at `u0_1 = [1.0, 1.0]` and fit it on its own. The
# initial conditions are stored in the experiment data via the `u0` keyword.

u0_1 = [1.0, 1.0]
prob_1 = ODEProblem(lotka_volterra!, u0_1, tspan, p_true)
M1 = reduce(hcat, solve(prob_1, Tsit5(); saveat = tsteps).u)
Y1 = M1 .+ 0.05 .* randn(rng, size(M1))
data_1 = ExperimentData(tsteps, Y1; σ = 0.05, u0 = u0_1)

pe_1 = ParameterEstimationLayer(prob_1, Tsit5(), data_1)
sol_1 = solve(problem(pe_1), LevenbergMarquardt(); abstol = 1.0e-8, reltol = 1.0e-8)
est_1 = estimate_parameters(pe_1, sol_1)
println("Fit from the first experiment only: ", sol_1.u)
println("Standard errors (first experiment): ", est_1.std_errors)

# ### 2. Second experiment with a different initial condition
# A second experiment is now run, starting from a *different* initial condition
# `u0_2 = [0.4, 2.5]`, observed on the same time grid.

u0_2 = [0.4, 2.5]
prob_2 = ODEProblem(lotka_volterra!, u0_2, tspan, p_true)
M2 = reduce(hcat, solve(prob_2, Tsit5(); saveat = tsteps).u)
Y2 = M2 .+ 0.05 .* randn(rng, size(M2))
data_2 = ExperimentData(tsteps, Y2; σ = 0.05, u0 = u0_2)

# ### 3. Joint fit over the combined data
#
# The parameters are shared between the experiments, but each experiment starts from its
# own initial conditions. An `ExperimentData` can hold *multiple* experiments: both are
# simulated with the **same** model -- the shooting layer of each experiment is built from
# the same problem with that experiment's initial conditions and observation times -- and
# the residuals of all experiments are concatenated into a single least-squares problem.

data_joint = ExperimentData(data_1, data_2)
println("Number of experiments: ", n_experiments(data_joint))
println("Total number of observations: ", n_observations(data_joint))

pe_joint = ParameterEstimationLayer(prob_1, Tsit5(), data_joint)

# To highlight the *incremental* nature of the refit, we warm-start the joint fit from the
# result of the first fit instead of the initial guess. The flat optimization variables of
# `pe_joint` contain the tunable *parameters only* -- here the four model parameters shared
# by both experiments, since the initial conditions are prescribed by the data. A warm
# start is therefore simply `sol_1.u`:

joint_prob = problem(pe_joint; x0 = sol_1.u)
sol_joint = solve(joint_prob, LevenbergMarquardt(); abstol = 1.0e-8, reltol = 1.0e-8)
est_joint = estimate_parameters(pe_joint, sol_joint)

println("First-experiment fit:   ", sol_1.u)
println("Joint fit:              ", sol_joint.u)
println("True parameters:        ", p_true)

# The joint fit uses both experiments, so the parameter uncertainties shrink relative to
# the first-experiment-only fit:
println("Standard errors (first experiment): ", est_1.std_errors)
println("Standard errors (joint fit):        ", est_joint.std_errors)

# The joint fit produces one fitted trajectory per experiment, which we can plot together
# with the observations of both runs.

result_joint = fitted_parameters(pe_joint, sol_joint)
traj_1, traj_2 = result_joint.trajectory

f = Figure()
ax = Axis(f[1, 1], xlabel = "t", ylabel = "population", title = "Joint fit over two experiments")
plot!(ax, traj_1, idxs = [1, 2])
plot!(ax, traj_2, idxs = [1, 2])
scatter!(ax, tsteps, Y1[1, :], markersize = 4, label = "exp 1: observed x")
scatter!(ax, tsteps, Y1[2, :], markersize = 4, label = "exp 1: observed y")
scatter!(ax, tsteps, Y2[1, :], markersize = 4, marker = :diamond, label = "exp 2: observed x")
scatter!(ax, tsteps, Y2[2, :], markersize = 4, marker = :diamond, label = "exp 2: observed y")
axislegend(ax, position = :rt)
display(f)
