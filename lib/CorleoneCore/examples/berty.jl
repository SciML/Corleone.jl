using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))


using CorleoneCore
using OrdinaryDiffEq
using SciMLSensitivity
using ComponentArrays
using LuxCore
using Random

using CairoMakie
using BenchmarkTools
using Zygote
using ForwardDiff

using Optimization
using OptimizationMOI
using Ipopt
using blockSQP
using LinearAlgebra

feed = [0.0; 0.0256; 0.1082; 0.3059; 0.0; 0.5603]
feed_ch3oh, feed_c02, feed_c0, feed_h2, feed_h20, feed_n2 = feed

T = 500.15
P = 50e5

beta_new = [ -6.2229, 42.2704, -4.60324, 21.204, -4.64011, 0.0290656, 0.370769, 0.0067666, 0.0441206, 0.847394, 0.00600975, 0.119349, 0.727213 ,1.30406]

G_new = [0.825972, -0.358255]
k_new = [ 0.00387321, 0.269224]
VGas_mkat_VN_new = [286*1e-6,  4.3501e-3, 3333/60*1e-6]

feed = [0.0; 0.0256; 0.1082; 0.3059; 0.0; 0.5603]
feed_ch3oh, feed_c02, feed_c0, feed_h2, feed_h20, feed_n2 = feed
p_new = vcat(T, P, feed , k_new, G_new, VGas_mkat_VN_new,  beta_new)

_protected_sqrt(x::T) where T = sqrt(x)

function berty(dy, y, p, t)

    T, P = p[1:2]
    #feed = [feed_ch3oh, feed_c02, feed_c0, feed_h2, feed_h20, feed_n2]
    feed = p[3:8]
    k = p[9:10] #[-0.0227896, 0.0598744]
    G1, G2 = p[11:12]
    V_Gas, m_kat, V_N = p[13:15]
    param = p[16:end]

    q_sat =  0.98
    R =  8.314472


    p_N = 1.01325e5
    #V_N = 3.33e-5 #m³/s Vollbrecht: 3.95e-6 m^3/s
    T_N =  273.15

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
    Pe = P / 1e5


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
    J21 = -param[14] * param[12] * Pe * y[2] * theta_dot^2 - param[13] * param[8] * Pe * y[2] * theta_star^ 2
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
        J11*dy[1]+J12*dy[2]+J13*dy[3]+J14*dy[4]+J15*dy[5],
        J21*dy[1]+J22*dy[2]+J23*dy[3]+J24*dy[4]+J25*dy[5],
        J31*dy[1]+J32*dy[2]+J33*dy[3],
        J41*dy[1]+J42*dy[2]+J44*dy[4]+J45*dy[5],
        J51*dy[1]+J52*dy[2]+J54*dy[4]+J55*dy[5],
        zero(eltype(dy))
    )
    y_lhs = vcat(
        nG *  dy[1:6] .+ (Pe * m_kat * q_sat) * (diagm(ones(6)) .- y[1:6]*ones(1,6)) * jacvec,
        dy[7]
    )

    # Reaction Rates
    k1_reac = exp(param[1] - param[2]*(T_MEAN/T-1));
    k2_reac = exp(param[3] - param[4]*(T_MEAN/T-1));
    k3_reac = exp(param[5] - param[6]*(T_MEAN/T-1));

    K_P1 = (10^(B1/T +A1 + log10(T)*C1 +D1*1e-3*T - E1*1e-7*T^2))
    K_P2 = (10^(A2/T +B2 + log10(T)*C2 +D2*T - E2*T^2))
    K_P3 = (10^(B3/T +A3 + log10(T)*C3 +D3*T + E3*T^2))

    # Surface Coverages
    eta_dot     = 1/(1+ param[11]*f_CO+param[12]*f_CH3OH+f_CO2*param[14])
    eta_circle  = 1/(1+param[7] *_protected_sqrt(f_H2))
    eta_star    = 1/(1+param[9]*f_H2O+param[8]*f_CH3OH+param[13]*f_CO2+param[9]*param[10]/param[7]^2 * f_H2O/f_H2)

    # Reaction Rates
    impact = [1 - y[7]; y[7]^2; y[7]/(1-y[7])]

    r1 = 0.33 * k1_reac * (Pe^3 * y[3]*y[4]^2 - (y[1]*Pe)/K_P1)                  * (eta_dot*eta_circle^4)
    r2 = 0.33 * k2_reac * (Pe^3 * y[2]*y[4]^2 - (y[1]*y[5]*Pe)/(K_P2*y[4]))      * (eta_star^2 * eta_circle^4)
    r3 = 0.33 * k3_reac * (Pe   * y[2]        - (y[3]* y[5]*Pe)/(K_P3* y[4]))    * (eta_star *eta_dot)
    y_rates =  impact .* [r1; r2; r3]

    # Component balances
    n_0 = (p_N * V_N) / (T_N * R)
    b_comp_bal, A_comp_bal = n_0 .* (feed .- y[1:6]) , m_kat .* (diagm(ones(6)) .- (y[1:6] * ones(1, 6))) * v
    # Catalyst dynamics
    K1 = exp(-G1 * 1e3 / (R * T))
    K2 = exp(-G2 * 1e3 / (R * T))

    dphi = ((k[1] * y[3] + k[2] * y[4]) * (0.9 - y[7]) - (1 / K1 * k[1] * y[2] + 1 / K2 * k[2] * y[5]) * y[7])
    y_rhs =  vcat(A_comp_bal*y_rates .+ b_comp_bal, dphi)
    return y_lhs .- y_rhs
end

u0 = [0.00854, 0.02494, 0.22906, 0.55062, 0.00184, 0.18499, 0.9];
du0 = zeros(7)
tspan = (0.0, 7200)

prob = DAEProblem(berty, du0, u0, tspan, p_new, differential_vars = ones(Bool, 7))#, initializealg = NoInit())

cgrid = collect(0.0:3600:7200)[1:end-1]

feed_co = ControlParameter(cgrid, name = :feed_co, controls = feed[2] * ones(length(cgrid)))
feed_co2 = ControlParameter(cgrid, name = :feed_co2, controls = feed[3] * ones(length(cgrid)))
feed_h2 = ControlParameter(cgrid, name = :feed_h2, controls = feed[4] * ones(length(cgrid)))
feed_n2 = ControlParameter(cgrid, name = :feed_n2, controls = feed[6] * ones(length(cgrid)))
cT = ControlParameter([0.0], name = :temperature, controls = [T])
V_N = ControlParameter(cgrid, name = :volume_flow, controls = last(VGas_mkat_VN_new) * ones(length(cgrid)))

layer = CorleoneCore.SingleShootingLayer(prob, DFBDF(),Int64[],[1,4,5,6,8,15], (cT,feed_co,feed_co2,feed_h2,feed_n2,V_N,))
ps, st = LuxCore.setup(Random.default_rng(), layer)
sol = solve(prob, DFBDF())
plot(sol)


sol, _ = layer(nothing, ps, st)

p = ComponentArray(ps)

oed_berty = CorleoneCore.augment_layer_for_oed(layer; params = [16], observed = (u,p,t) -> [u[1]; u[2]; u[3]], dt=900.0)

ps_oed, st_oed = LuxCore.setup(Random.default_rng(), oed_berty)

sol, _ = oed_berty(nothing, ps_oed, st_oed)
sol, _ = @btime oed_berty(nothing, ps_oed, st_oed)

f = Figure()
ax = CairoMakie.Axis(f[1,1], title="States")
ax1 = CairoMakie.Axis(f[2,1], title="Sensitivities")
ax2 = CairoMakie.Axis(f[1,2], title="FIM")
ax3 = CairoMakie.Axis(f[2,2])
[plot!(ax, sol.t,  _sol) for _sol in eachrow(Array(sol))[1:7]]
[plot!(ax1, sol.t, _sol) for _sol in eachrow(reduce(hcat, (sol[CorleoneCore.sensitivity_variables(oed_berty)])))]
[plot!(ax2, sol.t, _sol) for _sol in eachrow(reduce(hcat, (sol[CorleoneCore.fisher_variables(oed_berty)])))]
#stairs!(ax, control.t, (uopt + zero(p)).controls[1:length(control.t)])
#stairs!(ax3, control.t, (uopt + zero(p)).controls[length(control.t)+1:2*length(control.t)])
#stairs!(ax3, control.t, (uopt + zero(p)).controls[2*length(control.t)+1:3*length(control.t)])
f
