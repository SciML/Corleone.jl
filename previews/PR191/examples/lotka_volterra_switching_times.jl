using Corleone
using LuxCore
using Random
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using CairoMakie

function lotka_fishing!(du, u, p, t)
    T, w, c₁, c₂ = p
    x1, x2 = u[1], u[2]
    du[1] = T * (x1 - x1 * x2 - c₁ * w * x1)
    du[2] = T * (-x2 + x1 * x2 - c₂ * w * x2)
    du[3] = T * ((x1 - 1.0)^2 + (x2 - 1.0)^2)
    return nothing
end

tf = 12.0
u0 = [0.5, 0.7, 0.0]
catch_coefficients = [0.4, 0.2]

N = 9
pattern = [isodd(i) ? 0.0 : 1.0 for i in 1:N]

function make_stage(w, T0)
    prob = ODEProblem(
        lotka_fishing!, u0, (0.0, 1.0), [T0, w, catch_coefficients...];
        abstol = 1.0e-12, reltol = 1.0e-10
    )
    return ShootingLayer(
        prob, Tsit5(),
        Tunable(1; kind = Val(:parameter), lowerbound = 0.0, upperbound = tf) => nothing,
        Constant([2, 3, 4]; kind = Val(:parameter)) => nothing,
    )
end

carry_state = StageTransition(
    (traj, _) -> [last(traj.u)[1], last(traj.u)[2], 0.0]
)

stage_layers = [make_stage(pattern[i], tf / N) for i in 1:N]

layers = Any[]
for i in 1:N
    push!(layers, stage_layers[i])
    i < N && push!(layers, carry_state)
end

stages = Stages(layers...)

stage_names = [Symbol("stage_$(2i - 1)") for i in 1:N]
function stage_duration(ps, name)
    group = getproperty(ps, name).parameter.parameters.p₁
    return group.parameter.p₁.ps[1]
end

duration_constraint = (
    ps -> [sum(stage_duration(ps, name) for name in stage_names)]
) => (lb = [tf], ub = [tf])

loss = Tuple((:u₃, k) for k in 1:N)

optprob = OptimizationProblem(
    stages, AutoForwardDiff(), Val(:ComponentArrays);
    loss = loss, parameter_constraints = duration_constraint
)

uopt = solve(
    optprob, Ipopt.Optimizer();
    hessian_approximation = "limited-memory",
    max_iter = 500, tol = 1.0e-6, print_level = 0
)

ps, st = LuxCore.setup(Random.default_rng(), stages)
ps_opt = uopt + zero(ComponentArray(ps))
durations = [stage_duration(ps_opt, name) for name in stage_names]
(;
    objective = uopt.objective,
    relaxed_optimum = 1.34434,
    durations = round.(durations; digits = 4),
    total_duration = sum(durations),
)

function simulate(durations, pattern; saveat = 0.005)
    t_all = Float64[]
    x_all = Vector{Float64}[]
    w_all = Float64[]
    t0 = 0.0
    state = copy(u0)
    for i in eachindex(pattern)
        Ti = durations[i]
        prob = ODEProblem(
            lotka_fishing!, state, (t0, t0 + Ti), [1.0, pattern[i], catch_coefficients...]
        )
        sol = solve(prob, Tsit5(); saveat = saveat, abstol = 1.0e-11, reltol = 1.0e-11)
        for (t, u) in zip(sol.t, sol.u)
            push!(t_all, t)
            push!(x_all, u)
            push!(w_all, pattern[i])
        end
        state = sol.u[end]
        t0 += Ti
    end
    return t_all, x_all, w_all
end

t_phys, x_traj, fishing = simulate(durations, pattern)
x1 = [u[1] for u in x_traj]
x2 = [u[2] for u in x_traj]
cost = [u[3] for u in x_traj]

function plot_sto(t, x1, x2, cost, fishing, durations)
    switches = cumsum(durations[1:(end - 1)])
    f = Figure(size = (1000, 750))
    ax1 = CairoMakie.Axis(f[1, 1], title = "States", xlabel = "t", ylabel = "x")
    lines!(ax1, t, x1; linewidth = 2, label = "prey x₁")
    lines!(ax1, t, x2; linewidth = 2, label = "predator x₂")
    vlines!(ax1, switches; color = :black, linestyle = :dot, alpha = 0.4)
    axislegend(ax1; position = :rt)

    ax2 = CairoMakie.Axis(f[1, 2], title = "Control", xlabel = "t", ylabel = "w")
    stairs!(ax2, t, fishing; linewidth = 2, step = :post, label = "fishing w(t)")
    vlines!(ax2, switches; color = :black, linestyle = :dot, alpha = 0.4)
    axislegend(ax2; position = :rt)

    ax3 = CairoMakie.Axis(
        f[2, 1:2], title = "Accumulated cost", xlabel = "t", ylabel = "cost"
    )
    lines!(ax3, t, cost; linewidth = 2)
    return f
end

plot_sto(t_phys, x1, x2, cost, fishing, durations)

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
