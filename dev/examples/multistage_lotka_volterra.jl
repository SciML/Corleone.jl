using Corleone

using LuxCore
using Random
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using ForwardDiff
using CairoMakie
Makie.inline!()

function lotka_two_species(du, u, p, t)
    x, y, cost = u
    T, v, c₁, c₂ = p
    du[1] = T * (x - x * y - c₁ * v * x)
    du[2] = T * (-y + x * y - c₂ * v * y)
    du[3] = T * ((x - 1.0)^2 + (y - 1.0)^2)
    return nothing
end

function lotka_three_species(du, u, p, t)
    x, y, x3, cost = u
    T, v, c₁, c₂, c₃ = p
    du[1] = T * (x - x * y - c₁ * v * x)
    du[2] = T * (-y + x * y - c₂ * v * y)
    du[3] = T * (x3 * (y - x3 - c₃ * v))
    du[4] = T * ((x - 1.0)^2 + (y - 1.0)^2 + (x3 - 1.0)^2)
    return nothing
end

T1, T2 = 12.0, 12.0 # stage durations
η = 0.5             # initial population of the third species

prob1 = ODEProblem(
    lotka_two_species, [0.5, 0.7, 0.0], (0.0, 1.0), [T1, 0.0, 0.4, 0.2];
    abstol = 1.0e-10, reltol = 1.0e-9
)
prob2 = ODEProblem(
    lotka_three_species, [0.5, 0.7, η, 0.0], (0.0, 1.0), [T2, 0.0, 0.4, 0.2, 0.3];
    abstol = 1.0e-10, reltol = 1.0e-9
)

cgrid = collect(0.01:0.01:0.99)
function stage_parameters(T, n_parameters)
    return (
        Constant(1; kind = Val(:parameter), default = T) => nothing,
        Tunable(
            2; kind = Val(:parameter), default = 0.0, lowerbound = 0.0, upperbound = 1.0
        ) => cgrid,
        Constant(collect(3:n_parameters); kind = Val(:parameter)) => nothing,
    )
end
stage1 = ShootingLayer(
    prob1, Tsit5(), stage_parameters(T1, 4)...; controls = [2], quadratures = [3]
)
stage2 = ShootingLayer(
    prob2, Tsit5(), stage_parameters(T2, 5)...; controls = [2], quadratures = [4]
)

transition = StageTransition(
    (traj, _) -> begin
        xf = last(traj.u)
        [xf[1], xf[2], η, xf[3]]
    end
)
stages = Stages(stage1, transition, stage2)

optprob = OptimizationProblem(
    stages, AutoForwardDiff(), Val(:ComponentArrays);
    loss = :u₄,
)

uopt = solve(
    optprob, Ipopt.Optimizer();
    tol = 1.0e-8, hessian_approximation = "limited-memory",
    max_iter = 1000, print_level = 0
)
@assert successful_retcode(uopt)

ps, st = LuxCore.setup(Random.default_rng(), stages)
ps_opt = uopt + zero(ComponentArray(ps))
traj2, _ = stages(nothing, ps_opt, st)

v1 = Vector(ps_opt.stage_1.parameter.parameters.p₂.parameter.ps)
v2 = Vector(ps_opt.stage_3.parameter.parameters.p₂.parameter.ps)
(;
    objective = uopt.objective,
    superpredator_at_24 = last(traj2.u)[3],
    active_intervals_stage_1 = "$(count(>(0.01), v1)) / $(length(v1))",
    active_intervals_stage_2 = "$(count(>(0.01), v2)) / $(length(v2))",
)

stage1_dense = ShootingLayer(
    remake(prob1, saveat = 0.001), Tsit5(), stage_parameters(T1, 4)...;
    controls = [2], quadratures = [3]
)
stage2_dense = ShootingLayer(
    remake(prob2, saveat = 0.001), Tsit5(), stage_parameters(T2, 5)...;
    controls = [2], quadratures = [4]
)
stages_dense = Stages(stage1_dense, transition, stage2_dense)

_, st_dense = LuxCore.setup(Random.default_rng(), stages_dense)
traj_stage1, _ = stage1_dense(nothing, ps_opt.stage_1, st_dense.stage_1)
traj2_dense, _ = stages_dense(nothing, ps_opt, st_dense)

t1 = T1 .* traj_stage1.t
t2 = T1 .+ T2 .* traj2_dense.t
u1 = reduce(hcat, traj_stage1.u)
u2 = reduce(hcat, traj2_dense.u)

switch_times = vcat(T1 .* cgrid, T1 .+ T2 .* cgrid)
v_series = vcat(v1, v2)
v_full = v_series[
    clamp.(searchsortedlast.(Ref(switch_times), vcat(t1, t2)), 1, length(v_series)),
]

function plot_lotka(u1, t1, u2, t2, v_full)
    f = Figure(size = (1000, 750))
    c = Makie.wong_colors()
    ax1 = CairoMakie.Axis(
        f[1, 1], title = "Populations", xlabel = "t", ylabel = "population"
    )
    lines!(ax1, t1, u1[1, :]; linewidth = 2, label = "x (prey)", color = c[1])
    lines!(ax1, t1, u1[2, :]; linewidth = 2, label = "y (predator)", color = c[2])
    lines!(ax1, t2, u2[1, :]; linewidth = 2, color = c[1])
    lines!(ax1, t2, u2[2, :]; linewidth = 2, color = c[2])
    lines!(ax1, t2, u2[3, :]; linewidth = 2, label = "x3 (superpredator)", color = c[3])
    vlines!(ax1, T1, color = :black, linestyle = :dot)
    axislegend(ax1; position = :rt)

    axc = CairoMakie.Axis(
        f[2, 1], title = "Optimal fishing control", xlabel = "t", ylabel = "v(t)"
    )
    stairs!(axc, vcat(t1, t2), v_full; linewidth = 2, step = :pre)
    vlines!(axc, T1, color = :black, linestyle = :dot)

    axj = CairoMakie.Axis(
        f[2, 2], title = "Accumulated cost", xlabel = "t", ylabel = "J(t)"
    )
    lines!(axj, t1, u1[3, :]; linewidth = 2, color = c[5])
    lines!(axj, t2, u2[4, :]; linewidth = 2, color = c[5])
    vlines!(axj, T1, color = :black, linestyle = :dot)
    return f
end

plot_lotka(u1, t1, u2, t2, v_full)

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
