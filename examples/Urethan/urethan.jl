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

@parameters begin
    p1=1.0, [description = "Scaling parameter 1", tunable = false]
    p2=1.0, [description = "Scaling parameter 2", tunable = false]
    p3=1.0, [description = "Scaling parameter 3", tunable = false]
    p4=1.0, [description = "Scaling parameter 4", tunable = false]
    p5=1.0, [description = "Scaling parameter 5", tunable = false]
    p6=1.0, [description = "Scaling parameter 6", tunable = false]
end

# Variables for temperature and feed
@variables feed1(..)=0 [description = "State for feed 1", tunable=false, input=true, controlbounds=(0,0.0125)]
@variables feed2(..)=0 [description = "State for feed 2", tunable=false, input=true, controlbounds=(0,0.0125)]
@variables temperature(..)=373.15 [description = "State for temperature", tunable=false,
            input=true,  controlbounds = (-15.0, 15.0), bounds=(300.,400.)]


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
        0 ~ n_A0 + feed1(t) - n_C(t) - 2n_D(t) - 3n_E(t) - n_A(t);  #n_A
        0 ~ n_B0 + feed2(t) - n_C(t) - n_D(t) - n_B(t);   #n_B
        0 ~ n_L0 + (feed1(t) + feed2(t)) - n_L(t);     #n_L
    ], t, [n_A(t),n_B(t),n_C(t), n_D(t), n_E(t), n_L(t), feed1(t), feed2(t), temperature(t)],
     [p1, p2, p3, p4, p5, p6] , costs = Num[-∫(n_C(t))], consolidate=(x...) -> first(x)[1],
    #constraints = [temperature(ti) ≳ 300.0 for ti=10.0:10.0:80.0]
)

grid = ShootingGrid([0.,30.0, 50.0, 60.0])

control_points = collect(0.0:10.0:70.0)
N = length(control_points)

def_ut = [(-1)^i * 5.0 for i=1:N]#clamp.(5*randn(N), -15, 15)

controlmethod = DirectControlCallback(
    D(feed1(t)) => (; timepoints=control_points, defaults=zeros(N) .+ 0.0125),
    D(feed2(t)) => (; timepoints=control_points, defaults=zeros(N) .+ 0.0125),
    D(temperature(t)) => (; timepoints=control_points, defaults=def_ut),
)

builder = OCProblemBuilder(
    urethan, controlmethod, grid
)
#builder_ = builder()

#discrete_events(builder_.system)

optfun = OptimizationProblem{true}(builder, AutoForwardDiff(), Rodas5())

sol = optfun.f.f.predictor(optfun.u0)[1];# tspan=(-eps(),80.0))[1];
f = Figure()
ax = Axis(f[1,1], limits=((0,90), nothing), xticks=0:10:80)
plot!(ax, sol)
f

solve(optfun, Ipopt.Optimizer(); hessian_approximation = "limited-memory", max_iter = 50)

initialization_equations(builder_.system)

odeprob = ODEProblem(builder_.system, [], (0.0,80.0);
                check_compatibility=false, build_initializeprob=false)

sol = solve(odeprob, Rodas5(), u0=optfun.f.f.predictor.problem.u0)
plot(sol, idxs=[:temperature])
f = Figure()
ax = Axis(f[1,1], limits=(nothing, (-0.01, 0.015)))
plot!(ax, sol, idxs=[:feed1, :feed2])
f



using SciMLStructures
new_params = SciMLStructures.replace(SciMLStructures.Tunable(), optfun.f.f.predictor.problem.p, Corleone.invtransform(optfun.f.f.predictor.permutation, optfun.u0))
u0 = optfun.f.f.predictor.shooting_transition(optfun.f.f.predictor.problem.u0, new_params, 0.0)
sol = solve(optfun.f.f.predictor.problem, Rodas5(), tspan=(0, 80.0), u0=u0)
plot!(ax, sol)
f

using DifferentiationInterface

DifferentiationInterface.gradient(x-> optfun.f.f(x, nothing), AutoForwardDiff(), optfun.u0)
DifferentiationInterface.jacobian(x-> optfun.f.cons(x, nothing), AutoForwardDiff(), optfun.u0)

blocks = optfun.f.f.predictor.permutation.blocks

sol = solve(optfun, Ipopt.Optimizer(); max_iter = 100, tol = 1e-6,
         hessian_approximation="limited-memory", )

# Initial plot
sol = optfun.f.f.predictor(optfun.u0, saveat = 0.01)[1];
plot(sol, idxs = [:temperature])



p0 = CorleoneCore.get_p0(pred)
sol = pred(p0)
f = Figure()
ax1 = Axis(f[1,1], title="Temperature")
ax2 = Axis(f[2,1])
scatterlines!(ax1, sol.time, sol.states[1,:])
[scatterlines!(ax2, sol.time, x) for x in eachrow(sol.states[2:end,:])]
f