using Corleone
using LuxCore
using Random
using OrdinaryDiffEqTsit5
using Optimization
using OptimizationMOI
using Ipopt
using ComponentArrays
using CairoMakie

function stage1!(du, u, p, t)
    A, B, C, D, E, F, D_amount, V2 = u
    T1, T = p
    k1 = 0.0444 * exp(-2500.0 / T)
    k2 = 6889.0 * exp(-5000.0 / T)
    du[1] = T1 * (-2 * k1 * A^2)
    du[2] = T1 * (k1 * A^2 - k2 * B)
    du[3] = T1 * (k2 * B)
    du[4] = 0.0
    du[5] = 0.0
    du[6] = 0.0
    du[7] = 0.0 # amount of D stays constant (c_D = 0 throughout stage 1)
    du[8] = 0.0 # stage-2 volume is set at the switch
    return
end

function stage2!(du, u, p, t)
    A, B, C, D, E, F, D_amount, V2 = u
    T2 = p[1]
    du[1] = 0.0
    du[2] = T2 * (-0.02 * B - 0.05 * B - 0.00008 * B^2)
    du[3] = 0.0
    du[4] = T2 * (0.02 * B)
    du[5] = T2 * (0.05 * B)
    du[6] = T2 * (0.00004 * B^2)
    du[7] = T2 * (V2 * 0.02 * B) # d(D_amount)/dt = V2 * d(c_D)/dt
    du[8] = 0.0
    return
end

const V1 = 0.1
const c_b_s = 600.0

function switch_at_ts(x, θ)
    A, B, C, D, E, F, D_amount, V2 = x
    S = θ[1]
    V2_new = V1 + S
    return [
        V1 * A / V2_new, (V1 * B + S * c_b_s) / V2_new, V1 * C / V2_new,
        V1 * D / V2_new, V1 * E / V2_new, V1 * F / V2_new,
        V1 * D,    # amount of D is preserved across the switch
        V2_new,    # carry the stage-2 volume
    ]
end

u0 = [2000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

const t_p = 180.0   # processing time [min]
const c_d_w = 150.0 # minimum final concentration of D [mol/m^3]

T1_init = 5.0 * 15.0
T2_init = 15.0

prob1 = ODEProblem(stage1!, u0, (0.0, 1.0), [T1_init, 350.0], abstol = 1.0e-10)
prob2 = ODEProblem(stage2!, u0, (0.0, 1.0), [T2_init], abstol = 1.0e-10)

P = 20
grid = collect(range(0.0, 1.0, P + 1))[1:(end - 1)];

stage1 = ShootingLayer(
    prob1, Tsit5(),
    Tunable(1; kind = Val(:parameter), lowerbound = 10.0, upperbound = 170.0) => nothing,
    Tunable(2; kind = Val(:parameter), lowerbound = 298.0, upperbound = 398.0) => grid[2:end],
)
stage2 = ShootingLayer(
    prob2, Tsit5(),
    Tunable(1; kind = Val(:parameter), lowerbound = 10.0, upperbound = 170.0) => nothing,
)

let ps1, st1
    ps1, st1 = LuxCore.setup(Random.default_rng(), stage1)
    traj1, _ = stage1(nothing, ps1, st1)
    global u0_linkage = switch_at_ts(last(traj1.u), [0.1])
end

linkage = StageLinkage(
    switch_at_ts; u0 = u0_linkage, θ = [0.1], bounds_θ = ([0.0], [0.1])
)
stages = Stages(stage1, linkage, stage2)

duration_constraint = (
    ps -> [
        ps.stage_1.parameter.parameters.p₁.parameter.ps[1] +
            ps.stage_3.parameter.parameters.p₁.parameter.ps[1] - t_p,
    ]
) => (lb = [-Inf], ub = [0.0])

optprob = OptimizationProblem(
    stages, AutoForwardDiff(), Val(:ComponentArrays);
    loss = :(-u₇), # maximise J0 = V2 * c_D(t_P) = D_amount(t_P)
    constraints = (
        (:u₄, 2) => (t = [1.0], bounds = (c_d_w, Inf)),
        #(:u₂, 1) => (t= [1.0], bounds = (0.0, 2000.0)) # constraint on stage 1 (not needed)
    ),
    parameter_constraints = duration_constraint,
)

uopt = solve(
    optprob, Ipopt.Optimizer();
    hessian_approximation = "limited-memory",
    max_iter = 500, print_level = 5,
);

ps, st = LuxCore.setup(Random.default_rng(), stages)
ps_opt = uopt + zero(ComponentArray(ps))

println("J0 = V2*c_D(t_P) = ", -uopt.objective)
println("t_s = T1 = ", ps_opt.stage_1.parameter.parameters.p₁.parameter.ps[1], " min,  T2 = ", ps_opt.stage_3.parameter.parameters.p₁.parameter.ps[1], " min")
println("S = ", ps_opt.stage_2.θ[1], " m^3")
println("c_D(t_P) = ", last((stages(nothing, ps_opt, st))[1].u)[4], " mol/m^3")

stage1 = ShootingLayer(
    remake(prob1, saveat = 0.001), Tsit5(),
    Tunable(1; kind = Val(:parameter), lowerbound = 10.0, upperbound = 170.0) => nothing,
    Tunable(2; kind = Val(:parameter), lowerbound = 298.0, upperbound = 398.0) => grid[2:end],
)
stage2 = ShootingLayer(
    remake(prob2, saveat = 0.001), Tsit5(),
    Tunable(1; kind = Val(:parameter), lowerbound = 10.0, upperbound = 170.0) => nothing,
)

stages = Stages(stage1, linkage, stage2)
traj, _ = stages(nothing, ps_opt, st)
traj_stage1, _ = stage1(nothing, ps_opt.stage_1, st.stage_1)

T1 = ps_opt.stage_1.parameter.parameters.p₁.parameter.ps[1]
T2 = ps_opt.stage_3.parameter.parameters.p₁.parameter.ps[1]
t1p = T1 .* traj_stage1.t
t2p = T1 .+ T2 .* traj.t
u1 = reduce(hcat, traj_stage1.u)
u2 = reduce(hcat, traj.u)

temp_series = Vector(ps_opt.stage_1.parameter.parameters.p₂.parameter.ps)
temperature = temp_series[clamp.(searchsortedlast.(Ref(grid), traj_stage1.t), 1, length(temp_series))]

function plot_reactor(u1, t1p, u2, t2p, temperature)
    f = Figure(size = (1000, 750))
    c = CairoMakie.Makie.wong_colors()
    ax1 = CairoMakie.Axis(
        f[1, 1], title = "A, B, C",
        xlabel = "t [min]", ylabel = "c [mol/m³]",
        xticks = 0:30:180, yticks = 0:250:2000
    )
    for (i, n) in enumerate(("A", "B", "C"))
        lines!(ax1, t1p, u1[i, :]; linewidth = 2, label = n, color = c[i])
    end
    vlines!(ax1, T1, color = :black, linestyle = :dot)
    for (i, n) in enumerate(("A", "B", "C"))
        lines!(ax1, t2p, u2[i, :]; linewidth = 2, color = c[i])
    end
    axislegend(ax1; position = :rt)

    ax2 = CairoMakie.Axis(
        f[1, 2], title = "D, E, F",
        xlabel = "t [min]", ylabel = "c [mol/m³]",
        xticks = 0:30:180, yticks = 0:50:400
    )
    for (i, n) in enumerate(("D", "E", "F"))
        lines!(ax2, t2p, u2[3 + i, :]; linewidth = 2, label = n, color = c[3 + i])
    end
    for (i, n) in enumerate(("D", "E", "F"))
        lines!(ax2, t1p, u1[3 + i, :]; linewidth = 2, color = c[3 + i])
    end
    vlines!(ax2, T1, color = :black, linestyle = :dot)
    axislegend(ax2; position = :lt)

    axc = CairoMakie.Axis(
        f[2, 1:2], title = "Optimal temperature control",
        xlabel = "t [min]", ylabel = "T [K]",
        xticks = 0:30:180,
        limits = (nothing, (305, 345))
    )
    stairs!(axc, t1p, temperature; linewidth = 2, step = :pre, label = "T(t)")
    stairs!(axc, [extrema(t2p)...], ones(2) * last(temperature); linewidth = 2, step = :pre)
    vlines!(axc, T1, color = :black, linestyle = :dot)
    axislegend(axc; position = :rt)

    return f
end

plot_reactor(u1, t1p, u2, t2p, temperature)

        using Pkg
        Pkg.status()

        using InteractiveUtils
        InteractiveUtils.versioninfo()

        using Pkg
        Pkg.status(; mode = PKGMODE_MANIFEST)
