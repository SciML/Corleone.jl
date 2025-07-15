using Corleone
using TestEnv
TestEnv.activate()
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using OrdinaryDiffEq
using CairoMakie
using SciMLSensitivity, Optimization, OptimizationMOI, Ipopt
using SciMLSensitivity.ForwardDiff, SciMLSensitivity.Zygote, SciMLSensitivity.ReverseDiff
using blockSQP

const M    = [ 0.11911, 0.07412, 0.19323, 0.31234, 0.35733, 0.07806 ]
const rho  = [1095.0, 809.0, 1415.0, 1528.0, 1451.0, 1101.0]

Rg   = 8.314;
T1   = 363.16;

n_A0, n_B0, n_L0 = 0.1, 0.05, 0.01

# Set up variables
@variables n_C(..)=0.0  [tunable=false, description = "Molar numbers for C"]
@variables n_D(..)=0.0  [tunable=false, description = "Molar numbers for D"]
@variables n_E(..)=0.0  [tunable=false, description = "Molar numbers for E"]
@variables n_A(..)=n_A0 [tunable=false, irreducible = true, description = "Molar numbers for C"]
@variables n_B(..)=n_B0 [tunable=false, irreducible = true, description = "Molar numbers for D"]
@variables n_L(..)=n_L0 [tunable=false, irreducible = true, description = "Molar numbers for E"]
@variables h1(..)=0.0 [tunable = false, bounds=(0,1), description="Observed function 1"]
@variables h2(..)=0.0 [tunable = false, bounds=(0,1), description="Observed function 2"]
@variables h3(..)=0.0 [tunable = false, bounds=(0,1), description="Observed function 1"]

@parameters begin
    p1=1.0, [description = "Scaling parameter 1", uncertain=true, tunable = false]
    p2=1.0, [description = "Scaling parameter 2", uncertain=true, tunable = false]
    p3=1.0, [description = "Scaling parameter 3", tunable = false]
    p4=1.0, [description = "Scaling parameter 4", tunable = false]
    p5=1.0, [description = "Scaling parameter 5", tunable = false]
    p6=1.0, [description = "Scaling parameter 6", tunable = false]
end

# Variables for temperature and feed
@variables feed1(..)=0 [description = "State for feed 1", tunable=false,]
@variables feed2(..)=0 [description = "State for feed 2", tunable=false,]
@variables temperature(..)=373.15 [description = "State for temperature", tunable=false,
                bounds=(300.,400.)]

@variables df1(..) = 0.0125 [input=true, bounds = (0,0.0125)]
@variables df2(..) = 0.0125 [input=true, bounds = (0,0.0125)]
@variables dT(..) = 0.0 [input=true, bounds = (-15.0, 15.0)]

# Write system of equations
k_ref1    = p1 * 5.0E-4
E_a1      = p2 * 35240.0
k_ref2    = p3 * 8.0E-8
E_a2      = p4 * 85000.0
k_ref4    = p5 * 1.0E-8
E_a4      = p6 * 35000.0
dH_2      = -17031.0
K_C2      = 0.17

# Arrhenius equations for the reaction rates
fac_T = 1.0 / (Rg*temperature(t)) - 1.0 / (Rg*T1);
k1 = k_ref1 * exp(- E_a1 * fac_T);
k2 = k_ref2 * exp(- E_a2 * fac_T);
k4 = k_ref4 * exp(- E_a4 * fac_T);
K_C = K_C2 * exp(- dH_2 * fac_T);
k3 = k2/K_C;

# Reaction volume
V  = n_A(t) * M[1] / rho[1] + n_B(t) * M[2] /rho[2] + n_C(t) * M[3] / rho[3] + n_D(t) * M[4] / rho[4] +
            n_E(t) * M[5] / rho[5] + n_L(t) * M[6] / rho[6]

# Reaction rates
r1 = k1 * n_A(t)/V * n_B(t)/V;
r2 = k2 * n_A(t)/V * n_C(t)/V;
r3 = k3 * n_D(t)/V;
r4 = k4 * (n_A(t) / V)*(n_A(t) / V);

tspan  = (0.0, 80.0)
∫ = Symbolics.Integral(t in tspan)

sum_observed = n_A(t) * M[1]  + n_B(t) * M[2]  + n_C(t) * M[3]  + n_D(t) * M[4]  + n_E(t) * M[5] + n_L(t) * M[6]
# Define the eqs
@named urethan = ODESystem(
    [
        D(n_C(t)) ~ V * (r1 - r2 + r3); #n_C
        D(n_D(t)) ~ V * (r2 - r3);      #n_D
        D(n_E(t)) ~ V * r4;             #n_E
        D(feed1(t)) ~ df1(t);
        D(feed2(t)) ~ df2(t);
        D(temperature(t)) ~ dT(t);
        0 ~ n_A0 + feed1(t) - n_C(t) - 2n_D(t) - 3n_E(t) - n_A(t);  #n_A
        0 ~ n_B0 + feed2(t) - n_C(t) - n_D(t) - n_B(t);   #n_B
        0 ~ n_L0 + (feed1(t) + feed2(t)) - n_L(t);     #n_L
    ], t, [n_A(t),n_B(t),n_C(t), n_D(t), n_E(t), n_L(t), feed1(t), feed2(t), temperature(t), df1(t), df2(t), dT(t)],
     [p1, p2, p3, p4, p5, p6] ,
     observed = [h1(t) ~ 100 * n_C(t) / sum_observed; h2(t) ~ 100 * n_A(t) / sum_observed;
                    h3(t) ~ 100 * n_E(t) / sum_observed],
     costs = Num[-∫(n_C(t))], consolidate=(x...) -> first(x)[1],
    constraints = vcat(∫(h1(t)) ≲ 20.0, ∫(h2(t)) ≲ 20.0, ∫(h3(t)) ≲ 20.0,
                    [temperature(ti) ≳ 300.0 for ti=4.0:4.0:80.0],
                    [temperature(ti) ≲ 420.0 for ti=4.0:4.0:80.0])
)

grid = ShootingGrid([0.,40])

control_points = collect(0.0:4.0:80.0)[1:end-1]
N = length(control_points)

def_ut = [(-1)^i * 5.0 for i=1:N]#clamp.(5*randn(N), -15, 15)

controlmethod = DirectControlCallback(
    df1(t) => (; timepoints=control_points, defaults=zeros(N) .+ 0.0125),
    df2(t) => (; timepoints=control_points, defaults=zeros(N) .+ 0.0125),
    dT(t) => (; timepoints=control_points, defaults=def_ut),
    h1(t) => (; timepoints=control_points, defaults=ones(N)),
    h2(t) => (; timepoints=control_points, defaults=ones(N)),
    h3(t) => (; timepoints=control_points, defaults=ones(N))
)

builder = OEDProblemBuilder(
        urethan, controlmethod, grid, ACriterion(tspan)
    )

optfun = OptimizationProblem{true}(builder, AutoForwardDiff(), Rodas5())

sol_bsqp = solve(optfun, BlockSQPOpt(); maxiters = 50, options=blockSQP.sparse_options(),
                sparsity = optfun.f.f.predictor.permutation.blocks)


sol = optfun.f.f.predictor(sol_bsqp.u, saveat=1)[1];
f = Figure()
ax = Axis(f[1,1], limits=((0,80), nothing), title="States", xticks=0:10:80)
plot!(ax, sol, idxs=[:n_A, :n_B, :n_C])
f
sens_states = filter(Corleone.is_sensitivity, unknowns(optfun.f.f.predictor.problem.f.sys))
ax = Axis(f[1,2], limits=((0,80), nothing), title="Sensitivities", xticks=0:10:80)
plot!(ax, sol, idxs=Symbol.(operation.(sens_states)))
f
ax = Axis(f[2,2], limits=((0,80), nothing), title = "Sampling", xticks=0:10:80)
plot!(ax, sol, idxs=[:w1, :w2])
f
ax = Axis(f[2,1], limits=((0,80), nothing), xticks=0:10:80)
ax2 = Axis(f[2,1], limits=((0,80), nothing), yaxisposition=:right,
            yticklabelsvisible=false, xticklabelsvisible=false, xlabelvisible=false,
             xticks=0:10:80)
plot!(ax, sol, idxs=[:temperature], label="T")
f