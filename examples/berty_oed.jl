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

feed = [0.0; 0.0256; 0.2082; 0.6059; 0.0; 0.1603]
feed_ch3oh, feed_c02, feed_c0, feed_h2, feed_h20, feed_n2 = feed

T = 500.15
P = 50.0e5

beta_new = [-6.2229, 42.2704, -4.60324, 21.204, -4.64011, 0.0290656, 0.370769, 0.0067666, 0.0441206, 0.847394, 0.00600975, 0.119349, 0.727213, 1.30406]

G_new = [0.825972, -0.358255]
k_new = [0.00387321, 0.269224]
VGas_mkat_VN_new = [286 * 1.0e-6, 4.3501e-3, 3333 / 60 * 1.0e-6]

feed = [0.0; 0.0256; 0.1082; 0.3059; 0.0; 0.5603]
feed_ch3oh, feed_c02, feed_c0, feed_h2, feed_h20, feed_n2 = feed
p_new = vcat(T, P, feed, k_new, G_new, VGas_mkat_VN_new, beta_new)

_protected_sqrt(x::T) where {T} = sqrt(x)

function berty(dy, y, p, t)

    T, P = p[1:2]
    #feed = [feed_ch3oh, feed_c02, feed_c0, feed_h2, feed_h20, feed_n2]
    feed = p[3:8]
    k = p[9:10] #[-0.0227896, 0.0598744]
    G1, G2 = p[11:12]
    V_Gas, m_kat, V_N = p[13:15]
    param = p[16:end]

    q_sat = 0.98
    R = 8.314472


    p_N = 1.01325e5
    #V_N = 3.33e-5 #m³/s Vollbrecht: 3.95e-6 m^3/s
    T_N = 273.15

    T_MEAN = 523.15

    A1 = 13.814
    B1 = 3784.7
    C1 = -9.2833
    D1 = 3.1475
    E1 = 4.2613
    A2 = 1581.7
    B2 = 15.0921
    C2 = -8.7639
    D2 = +2.1105e-3
    E2 = 1.9303e-7
    A3 = 1.2777
    B3 = -2167.0
    C3 = 0.5194
    D3 = -1.037e-3
    E3 = 2.331e-7

    v = [1 1 0; 0 -1 -1; -1 0 1; -2 -3 -1; 0 1 1; 0 0 0]


    # LHS
    nG = P * V_Gas / (R * T)
    Pe = P / 1.0e5


    f_CH3OH = Pe * y[1]
    f_CO2 = Pe * y[2]
    f_CO = Pe * y[3]
    f_H2 = Pe * y[4]
    f_H2O = Pe * y[5]

    theta_dot = 1 / (1 + param[11] * f_CO + param[12] * f_CH3OH + f_CO2 * param[14])
    theta_star = 1 / (1 + param[9] * f_H2O + param[8] * f_CH3OH + param[13] * f_CO2 + param[10] * param[9] / param[7]^2 * f_H2O / f_H2)
    theta_circle = 1 / (1 + param[7] * _protected_sqrt(f_H2))

    # First column
    J11 = param[12] * theta_dot - param[12]^2 * Pe * y[1] * theta_dot^2 + param[8] * theta_star - param[8]^2 * Pe * y[1] * theta_star^2
    J12 = -param[14] * param[12] * Pe * y[1] * theta_dot^2 - param[8] * param[13] * Pe * y[1] * theta_star^2
    J13 = -param[11] * param[12] * Pe * y[1] * theta_dot^2
    J14 = param[8] * param[9] * param[10] / param[7]^2 * y[5] / (Pe * y[4]^2) * Pe * y[1] / Pe * theta_star^2
    J15 = -param[8] * param[9] * (1 + param[10] / param[7]^2 * 1 / (Pe * y[4])) * Pe * y[1] * theta_star^2


    # dThtheta_dot_CO2/dy_i
    J21 = -param[14] * param[12] * Pe * y[2] * theta_dot^2 - param[13] * param[8] * Pe * y[2] * theta_star^2
    J22 = param[14] * theta_dot - param[14]^2 * Pe * y[2] * theta_dot^2 + param[13] * theta_star - param[13]^2 * Pe * y[2] * theta_star^2
    J23 = -param[11] * param[14] * Pe * y[2] * theta_dot^2
    J24 = param[13] * param[9] * param[10] / param[7]^2 * y[5] / (Pe * y[4]^2) * Pe * y[2] * theta_star^2
    J25 = -param[13] * param[9] * (1 + param[10] / param[7]^2 * 1 / (Pe * y[4])) * Pe * y[2] * theta_star^2

    # dThtheta_dot_CO/dy_i
    J31 = -param[11] * param[12] * Pe * y[3] * theta_dot^2
    J32 = -param[14] * param[11] * Pe * y[3] * theta_dot^2
    J33 = param[11] * theta_dot - param[11]^2 * Pe * y[3] * theta_dot^2

    J41 = -param[7] * param[8] * _protected_sqrt(Pe * y[4]) * theta_star^2
    J42 = -param[7] * param[13] * _protected_sqrt(Pe * y[4]) * theta_star^2
    J44 = 0.5 * param[7] * 1 / _protected_sqrt(Pe * y[4]) * theta_star + param[9] * param[10] / param[7]^2 * (Pe * y[5] / (Pe * y[4])^2) * theta_star^2 + 0.5 * (param[7] * 0.5 / _protected_sqrt(Pe * y[4])) * theta_circle - 0.5 * param[7]^2 * theta_circle^2
    J45 = -param[7] * _protected_sqrt(Pe * y[4]) * param[9] * (1 + param[10] / param[7]^2 * 1 / (Pe * y[4])) * theta_star^2

    J51 = -param[9] * param[8] * Pe * y[5] * theta_star^2
    J52 = -param[9] * param[13] * Pe * y[5] * theta_star^2
    J54 = param[9] * param[9] * param[10] / param[7]^2 * Pe * y[5] / ((Pe * y[4])^2) * Pe * y[5] * theta_star^2
    J55 = param[9] * theta_star - param[9]^2 * (1 + param[10] / param[7]^2 * 1 / (Pe * y[4])) * Pe * y[5] * theta_star^2

    jacvec = vcat(
        J11 * dy[1] + J12 * dy[2] + J13 * dy[3] + J14 * dy[4] + J15 * dy[5],
        J21 * dy[1] + J22 * dy[2] + J23 * dy[3] + J24 * dy[4] + J25 * dy[5],
        J31 * dy[1] + J32 * dy[2] + J33 * dy[3],
        J41 * dy[1] + J42 * dy[2] + J44 * dy[4] + J45 * dy[5],
        J51 * dy[1] + J52 * dy[2] + J54 * dy[4] + J55 * dy[5],
        zero(eltype(dy))
    )
    y_lhs = vcat(
        nG * dy[1:6] .+ Pe * m_kat * q_sat * (diagm(ones(6)) .- y[1:6] * ones(1, 6)) * jacvec,
        dy[7]
    )

    # Reaction Rates
    k1_reac = exp(param[1] - param[2] * (T_MEAN / T - 1))
    k2_reac = exp(param[3] - param[4] * (T_MEAN / T - 1))
    k3_reac = exp(param[5] - param[6] * (T_MEAN / T - 1))

    K_P1 = (10^(B1 / T + A1 + log10(T) * C1 + D1 * 1.0e-3 * T - E1 * 1.0e-7 * T^2))
    K_P2 = (10^(A2 / T + B2 + log10(T) * C2 + D2 * T - E2 * T^2))
    K_P3 = (10^(B3 / T + A3 + log10(T) * C3 + D3 * T + E3 * T^2))

    # Surface Coverages
    eta_dot = 1 / (1 + param[11] * f_CO + param[12] * f_CH3OH + f_CO2 * param[14])
    eta_circle = 1 / (1 + param[7] * _protected_sqrt(f_H2))
    eta_star = 1 / (1 + param[9] * f_H2O + param[8] * f_CH3OH + param[13] * f_CO2 + param[9] * param[10] / param[7]^2 * f_H2O / f_H2)

    # Reaction Rates
    impact = [1 - y[7]; y[7]^2; y[7] / (1 - y[7])]

    r1 = 0.33 * k1_reac * (Pe^3 * y[3] * y[4]^2 - (y[1] * Pe) / K_P1) * (eta_dot * eta_circle^4)
    r2 = 0.33 * k2_reac * (Pe^3 * y[2] * y[4]^2 - (y[1] * y[5] * Pe) / (K_P2 * y[4])) * (eta_star^2 * eta_circle^4)
    r3 = 0.33 * k3_reac * (Pe * y[2] - (y[3] * y[5] * Pe) / (K_P3 * y[4])) * (eta_star * eta_dot)
    y_rates = impact .* [r1; r2; r3]

    # Component balances
    n_0 = (p_N * V_N) / (T_N * R)
    b_comp_bal, A_comp_bal = n_0 .* (feed .- y[1:6]), m_kat .* (diagm(ones(6)) .- (y[1:6] * ones(1, 6))) * v
    # Catalyst dynamics
    K1 = exp(-G1 * 1.0e3 / (R * T))
    K2 = exp(-G2 * 1.0e3 / (R * T))

    dphi = ((k[1] * y[3] + k[2] * y[4]) * (0.9 - y[7]) - (1 / K1 * k[1] * y[2] + 1 / K2 * k[2] * y[5]) * y[7])
    y_rhs = vcat(A_comp_bal * y_rates .+ b_comp_bal, dphi)
    return y_lhs .- y_rhs
end

u0 = [0.00854, 0.02494, 0.22906, 0.55062, 0.00184, 0.18499, 0.9];
du0 = zeros(7)
tspan = (0.0, 7200.0)

prob = DAEProblem{false}(
    berty, du0, u0, tspan, p_new,
    initializealg = BrownFullBasicInit(), #CheckInit(),
    differential_vars = ones(Bool, 7)
)
#, initializealg = BrownFullBasicInit())

sol = solve(prob, DFBDF(), dense = true, progress = true)
init_dae = init(prob, DFBDF())

prob = DAEProblem{false}(
    berty, init_dae.du, init_dae.u, tspan, p_new,
    initializealg = NoInit(),
    #abstol=1e-8, reltol=1e-6,
    sensealg = AutoFiniteDiff(),
    differential_vars = ones(Bool, 7)
)
sol = solve(prob, DFBDF())
plot(sol)

@btime solve(prob, DFBDF())
## Optimize only sampling times with all else fixed
layer = OEDLayer(prob, DFBDF(); params = [16], observed = (u, p, t) -> u[1:1]);

ps, st = LuxCore.setup(Random.default_rng(), layer)
pc = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer)
layer.layer.problem.f(layer.layer.problem.du0, layer.layer.problem.u0, layer.layer.problem.p, 0.0)

sol = solve(layer.layer.problem, DFBDF())
plot(sol)

@btime sols, _ = layer(nothing, pc, st)

f = Figure()
ax = CairoMakie.Axis(f[1, 1])
[plot!(ax, sols.t, solu) for solu in eachrow(Array(sols))]
f

crit = ACriterion()
ACrit = crit(layer)
ACrit(pc, nothing)


sampling_cons = let ax = getaxes(pc), dt = diff(first(layer.layer.controls).t)[1]
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= sum(ps.controls) * dt
    end
end

optfun = OptimizationFunction(
    ACrit, AutoForwardDiff(), cons = sampling_cons
)

optprob = OptimizationProblem(
    optfun, collect(pc), lb = collect(lb), ub = collect(ub), lcons = zeros(1), ucons = [360.0]
)

uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 1.0e-10,
    hessian_approximation = "limited-memory",
    max_iter = 300
)


IG = InformationGain(layer, uopt)
multiplier = uopt.original.inner.mult_g

f = Figure()
ax = CairoMakie.Axis(f[1, 1], title = "States")
ax1 = CairoMakie.Axis(f[2, 1], title = "Sensitivities")
ax2 = CairoMakie.Axis(f[1, 2], title = "Information Gain")
ax3 = CairoMakie.Axis(f[2, 2], title = "Sampling")
[plot!(ax, sol.t, _sol) for _sol in eachrow(Array(sol))[1:7]]
[plot!(ax1, sol.t, _sol) for _sol in eachrow(reduce(hcat, (sol[Corleone.sensitivity_variables(layer)[:]])))]
plot!(ax2, IG.t, tr.(IG.global_information_gain[1]))
hlines!(ax2, multiplier)
stairs!(ax3, first(layer.layer.controls).t, (uopt + zero(pc)).controls)
f

## Optimize temperature and feeds along with sampling times

min_n2(x, P, V_N) = -x + (8.0 * P / 1.0e5) / (V_N * 60 / 1.0e-6)

min_N2 = min_n2(0.0, P, 3333.0 / 60 * 1.0e-6)

control = ControlParameter(0.0:600:6600.0, name = :temperature, controls = T * ones(12), bounds = (470.0, 530.0))

co_feed = ControlParameter(0.0:900:6300.0, name = :feed_co, controls = feed_c0 * ones(8), bounds = (0.0, 0.4))
co2_feed = ControlParameter(0.0:900:6300.0, name = :feed_co2, controls = feed_c02 * ones(8), bounds = (0.0, 0.4))
h2_feed = ControlParameter(0.0:900:6300.0, name = :feed_h2, controls = feed_h2 * ones(8), bounds = (0.0, 0.75))
n2_feed = ControlParameter(0.0:900:6300.0, name = :feed_n2, controls = feed_n2 * ones(8), bounds = (min_N2, 1.0))

layer = OEDLayer(
    prob, DFBDF(); params = [18, 20], observed = (u, p, t) -> u[1:4],
    controls = (control, co2_feed, co_feed, h2_feed, n2_feed), control_indices = [1, 4, 5, 6, 8]
);
ps, st = LuxCore.setup(Random.default_rng(), layer)
pc = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer)

nc = map(x -> length(x.t), layer.layer.controls)


sampling_cons_with_T = let ax = getaxes(pc), dt = diff(layer.layer.controls[end].t)[1]
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        feed = [ps.controls[12 + i] + ps.controls[20 + i] + ps.controls[28 + i] + ps.controls[36 + i] for i in 1:8]
        sampling = [sum(ps.controls[(44 + (i - 1) * 100 + 1):(44 + i * 100)]) * dt for i in 1:layer.dimensions.nh]
        res .= vcat(feed, sampling)
    end
end

sampling_cons_with_T(zeros(12), pc, nothing)

using LinearAlgebra
using SparseArrays
using ForwardDiff
J = ForwardDiff.jacobian(x -> sampling_cons_with_T(zeros(eltype(x), 12), x, nothing), collect(pc))
spy(J)
sparse_J = sparse(J)


optfun = OptimizationFunction(
    ACriterion()(layer), AutoForwardDiff(), cons = sampling_cons_with_T,
    cons_jac_prototype = sparse_J
)

lbc = vcat(ones(8), zeros(4))
ubc = vcat(ones(8), zeros(4) .+ 3600)

optprob = OptimizationProblem(
    optfun, collect(ComponentArray(ps)), lb = collect(lb),
    ub = collect(ub), lcons = lbc, ucons = ubc
)

optprob = remake(optprob, u0 = uopt.u)
uopt = solve(
    optprob, Ipopt.Optimizer(),
    tol = 5.0e-6,
    hessian_approximation = "limited-memory",
    max_iter = 50
)

Corleone.fim(layer, uopt)
sols, _ = layer(nothing, uopt + zero(pc), st)
IG = InformationGain(layer, uopt)
multiplier = uopt.original.inner.mult_g

f = Figure()
ax = CairoMakie.Axis(f[1, 1], title = "States")
ax1 = CairoMakie.Axis(f[2, 1], title = "Sensitivities")
ax2 = CairoMakie.Axis(f[1, 2], title = "Information Gain")
ax3 = CairoMakie.Axis(f[2, 2], title = "Sampling")
ax4 = CairoMakie.Axis(f[3, 1], title = "Temperature")
ax5 = CairoMakie.Axis(f[3, 2], title = "Feed")
[plot!(ax, sols.t, _sol) for _sol in eachrow(Array(sols))[1:7]]
[plot!(ax1, sols.t, _sol) for _sol in eachrow(reduce(hcat, (sols[Corleone.sensitivity_variables(layer)[:]])))]
[plot!(ax2, IG.t, tr.(GIG)) for GIG in IG.global_information_gain]
hlines!(ax2, multiplier)
stairs!(ax4, layer.layer.controls[1].t, (uopt + zero(pc)).controls[1:12])
[stairs!(ax5, layer.layer.controls[2].t, (uopt + zero(pc)).controls[(13 + (i - 1) * 8):(12 + i * 8)]) for i in 1:4]
[stairs!(ax3, layer.layer.controls[end].t, (uopt + zero(pc)).controls[(44 + (i - 1) * 100 + 1):(44 + i * 100)]) for i in 1:layer.dimensions.nh]
f
