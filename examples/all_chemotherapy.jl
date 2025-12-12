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

using CSV
using DataFrames

# Individual model parameters
df = CSV.read("data/all_chemotherapy.csv", DataFrame)
mask = df.ID .== 1150 

myANC     = df[mask, :myANC][1] 
myu       = df[mask, :myu][1] 
Vml_6MP   = df[mask, :Vml_MP][1]
Kml_6MP   = df[mask, :Kml_MP][1]
Keff_6MP  = df[mask, :Keff_MP][1]
base      = df[mask, :Base_x][1]
ktr       = df[mask, :ktr][1]
slope_TGN = df[mask, :slope_TGN][1]
gamma     = df[mask, :gamma][1]
Anc_INI   = df[mask, :Anc_INI][1]
INITGN    = df[mask, :INITGN][1]

# Constants
kaMP = 21.07    # [1/day]
keMP = 15.4     # [1/day]
V_MP = 20.1911  # [L/m^2]
kcirc = 2.3765
unitconsistency = 10^6 / 152180 * 0.12

# Initial values
u0 = [0., 0., INITGN, 
      Anc_INI*kcirc/ktr, Anc_INI*kcirc/ktr, Anc_INI*kcirc/ktr, Anc_INI*kcirc/ktr, 
      Anc_INI, 0.0]


function f(du, u, p, t)
    x_MP = u[1]
    x_MP_Plasma = u[2]
    x_E6MP = u[3]
    x_Prol = u[4]
    x_tr1 = u[5]
    x_tr2 = u[6]
    x_tr3 = u[7]
    x_ma = u[8]
    
    du[1] = -kaMP * x_MP
    du[2] = kaMP * x_MP - keMP * x_MP_Plasma
    du[3] = (Vml_6MP * x_MP_Plasma / V_MP) / (Kml_6MP + x_MP_Plasma / V_MP) - Keff_6MP * x_E6MP
    du[4] = ktr * x_Prol * (1 - slope_TGN * x_E6MP) * (base / x_ma)^gamma - ktr * x_Prol
    du[5] = ktr * (x_Prol - x_tr1)
    du[6] = ktr * (x_tr1 - x_tr2)
    du[7] = ktr * (x_tr2 - x_tr3)
    du[8] = ktr * x_tr3 - kcirc * x_ma
    du[9] = (x_ma - myANC)^2
end

dosetimes = vcat(0.0, 1:1:149)
# Callback that stops integration when one of the dosetimes is hit..
condition(u, t, integrator) = t ∈ dosetimes
# .. and then increases the value of u[1] by the value of p[1].
affect!(integrator) = integrator.u[1] += integrator.p[1] * unitconsistency
cb = DiscreteCallback(condition, affect!)

dosetimes2 = vcat(150.0, 151:1:364)
# Callback that stops integration when one of the dosetimes is hit..
condition2(u, t, integrator) = t ∈ dosetimes2
# .. and then increases the value of u[1] by the value of p[1].
affect2!(integrator) = integrator.u[1] += myu * unitconsistency
cb2 = DiscreteCallback(condition2, affect2!)

prob = ODEProblem(f, u0, (0.0, 365.0), [1.0], callback=CallbackSet(cb,cb2), tstops=vcat(dosetimes, dosetimes2))

#sol = solve(prob, Tsit5())
#plot(sol, idxs=[1,2,3,4,5,6])


# p[1] is then specified to be our control
control = ControlParameter(
    dosetimes, name = :c, bounds=(0.0,150.0), controls = 100.0*ones(length(dosetimes))
)

# Set up layer that applies the controls to the problem
layer = SingleShootingLayer(prob, Tsit5(), [1], (control,))
ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer)

sol, _ = layer(nothing, p, st)

objective = let layer = layer, st = st, ax = getaxes(p)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        last(sols.u)[9]
    end
end
objective(collect(p), nothing)

optfun = OptimizationFunction(
    objective, AutoForwardDiff(),
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub)
)

uopt = solve(optprob, Ipopt.Optimizer(),
    tol = 1e-6,
    hessian_approximation = "limited-memory",
    max_iter = 100
)

sol, _ = layer(nothing, uopt + zero(p), st)

fig = Figure()
ax = CairoMakie.Axis(fig[1,1], title="States")
[lines!(ax, sol.t, reduce(hcat, sol.u)[i,:]) for i=8:8]
hlines!(ax, [myANC], linestyle=:dash, color=:gray)
ax1 = CairoMakie.Axis(fig[2,1], title="Dosage", xlabel="Time t")
barplot!(ax1, dosetimes[2:end], uopt.u[1:end-1], width=.2)
barplot!(ax1, dosetimes2, myu*ones(length(dosetimes2)), width=.2, color=:gray)
CairoMakie.linkxaxes!(ax,ax1)
display(fig)
