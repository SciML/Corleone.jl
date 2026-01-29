using Pkg
Pkg.activate(@__DIR__)
using Corleone
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using LuxCore
using Random

using CairoMakie
using Optimization
using OptimizationMOI
using Ipopt
using blockSQP
using LinearAlgebra

tspan = (0.0, 80.0)

const M = [0.11911, 0.07412, 0.19323, 0.31234, 0.35733, 0.07806]
const rho = [1095.0, 809.0, 1415.0, 1528.0, 1451.0, 1101.0]

Rg = 8.314;
T1 = 363.16;

n_A0, n_B0, n_L0 = 0.1, 0.05, 0.01

function urethan(u, p, t)
    na, nb, nc, nd, ne, nl, T = u

    p_sca = p[1:6]        # parameters
    c_T, c1, c2 = p[7:9]  # controls
    k_ref1 = p_sca[1] * 5.0e-4
    E_a1 = p_sca[2] * 35240.0
    k_ref2 = p_sca[3] * 8.0e-8
    E_a2 = p_sca[4] * 85000.0
    k_ref4 = p_sca[5] * 1.0e-8
    E_a4 = p_sca[6] * 35000.0
    dH_2 = -17031.0
    K_C2 = 0.17

    Rg = 8.314
    T1 = 363.16

    fac_T = 1.0 / (Rg * T) - 1.0 / (Rg * T1)
    k1 = k_ref1 * exp(- E_a1 * fac_T)
    k2 = k_ref2 * exp(- E_a2 * fac_T)
    k4 = k_ref4 * exp(- E_a4 * fac_T)
    K_C = K_C2 * exp(- dH_2 * fac_T)
    k3 = k2 / K_C
    V = na * M[1] / rho[1] + nb * M[2] / rho[2] + nc * M[3] / rho[3] +
        nd * M[4] / rho[4] + ne * M[5] / rho[5] + nl * M[6] / rho[6]

    r1 = k1 * na / V * nb / V
    r2 = k2 * na / V * nc / V
    r3 = k3 * nd / V
    r4 = k4 * (na / V) * (na / V)

    return [
        V * (-r1 - r2 + r3 - 3 * r4) + c1;
        V * (-r1) + c2;
        V * (r1 - r2 + r3);
        V * (r2 - r3);
        V * r4;
        c1 + c2;
        c_T
    ]
end


function observed(u, p, t)
    na, nb, nc, nd, ne, nl, T = u
    sum_observed = na * M[1] + nb * M[2] + nc * M[3] + nd * M[4] + ne * M[5] + nl * M[6]
    return [
        100 * na * M[1] / sum_observed;
        100 * nc * M[3] / sum_observed;
        #100 * nd*M[4]/sum_observed;
        100 * ne * M[5] / sum_observed
    ]
end

f_urethan = ODEFunction(urethan; observed = observed) #, syms=Symbol.(["na"; "nb"; "nc"; "nd"; "ne"; "nl"; "T"]))

u0 = [0.1, 0.05, 0.01, 0.0, 0.0, 0.0, 300] #293.15]

p = vcat(ones(6), 1.25, 0.0125, 0.0125)
prob_normal = ODEProblem(f_urethan, u0, tspan, p)
sol = solve(prob_normal, Tsit5())
plot(sol, idxs = [1, 2, 3, 4, 5, 6])


control_points = collect(0.0:8.0:80.0)[1:(end - 1)]

c1 = ControlParameter(
    control_points, name = :c1, controls = 0.0125 * ones(length(control_points)), bounds = (0.0, 0.0125)
)
c2 = ControlParameter(
    control_points, name = :c2, controls = 0.0125 * ones(length(control_points)), bounds = (0.0, 0.0125)
)
cT = ControlParameter(
    control_points, name = :cT, controls = [(-1)^i * 5.0 for i in 1:length(control_points)], bounds = (-5.0, 5.0)
)

layer = Corleone.SingleShootingLayer(prob_normal, Tsit5(), [7, 8, 9], (cT, c1, c2))
ps, st = LuxCore.setup(Random.default_rng(), layer)

sol = layer(nothing, ps, st)

oed_layer = Corleone.augment_layer_for_oed(layer; params = [1, 2], observed = prob_normal.f.observed)
psoed, stoed = LuxCore.setup(Random.default_rng(), oed_layer)
sol_oed, _ = oed_layer(nothing, psoed, stoed)
p = ComponentArray(psoed)
lb, ub = Corleone.get_bounds(oed_layer)

loss = let layer = oed_layer, st = stoed, ax = getaxes(p), crit = ACriterion()
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        crit(layer, sols)
    end
end


loss(collect(p), nothing)
nc = length(c1.t)
sampling_cons = let layer = oed_layer, st = stoed, nc = nc, dt = diff(last(layer.controls).t)[1], ax = getaxes(p)
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= [
            sum(ps.controls[(3 * nc + 1):(4 * nc)]) * dt;
            sum(ps.controls[(4 * nc + 1):(5 * nc)]) * dt;
            sum(ps.controls[(5 * nc + 1):(6 * nc)]) * dt
        ]
    end
end

optfun = OptimizationFunction(
    loss, AutoForwardDiff(), cons = sampling_cons
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub), lcons = zeros(3), ucons = [20.0, 20.0, 20.0]
)

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 300
)

sol_oed, _ = oed_layer(nothing, uopt + zero(p), stoed)

f = Figure()
ax = CairoMakie.Axis(f[1, 1])
ax1 = CairoMakie.Axis(f[2, 1])
ax2 = CairoMakie.Axis(f[1, 2])
ax3 = CairoMakie.Axis(f[2, 2])
ax4 = CairoMakie.Axis(f[3, 2])
[plot!(ax, sol_oed.t, sol) for sol in eachrow(Array(sol_oed))[1:6]]
[plot!(ax1, sol_oed.t, sol) for sol in eachrow(reduce(hcat, (sol_oed[Corleone.sensitivity_variables(oed_layer)[:]])))]
[plot!(ax2, sol_oed.t, sol) for sol in eachrow(reduce(hcat, (sol_oed[Corleone.fisher_variables(oed_layer)])))]
stairs!(ax3, c1.t, (uopt + zero(p)).controls[1:length(c1.t)])
stairs!(ax3, c2.t, (uopt + zero(p)).controls[(length(c1.t) + 1):(2 * length(c1.t))])
stairs!(ax3, cT.t, (uopt + zero(p)).controls[(2 * length(c1.t) + 1):(3 * length(c1.t))])
stairs!(ax4, c1.t, (uopt + zero(p)).controls[(3 * length(c1.t) + 1):4length(c1.t)])
stairs!(ax4, c2.t, (uopt + zero(p)).controls[(4length(c1.t) + 1):(5 * length(c1.t))])
stairs!(ax4, cT.t, (uopt + zero(p)).controls[(5 * length(c1.t) + 1):(6 * length(c1.t))])

f
