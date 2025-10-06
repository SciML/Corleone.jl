using Pkg
Pkg.activate(joinpath(@__DIR__, "../"))
using Corleone
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

M = [0.1362, 0.09806, 0.23426, 0.236] # molar masses
R = 8.314 # general gas constant
T_ref = 293 # reference temperature

k1 = 0.01
k_kat = 0.1
E1 = 6.e4
E_kat = 4.e4
lamb = 0.25

p_values = [k1, k_kat, E1, E_kat, lamb]
#p_scale = ones(5)

function diels_alder(u,p,t)
    m_total = u[1] * M[1] + u[2] * M[2] + u[4] * M[4]

    p_scaled = p[1:5] #.* p_values

    T = p[6] + 273.
    k = p_scaled[1] * exp(-p_scaled[3] * (1 / T - 1 / T_ref)) + p_scaled[2] * p[7] * exp(-p_scaled[5] * t) * exp(-p_scaled[4] / R * (1 / T - 1 / T_ref))
    return  [-k * (u[1] * u[2]) / m_total,
                            -k * (u[1] * u[2]) / m_total,
                            k * (u[1] * u[2]) / m_total,
                            0]
end

c_kat = 0.1

#p = vcat(p_scale, 20.0, c_kat)
p = vcat(p_values, 20.0, c_kat) # T, c_kat
u0 = [1.0, 1.0, 0.0, 2.0]

tspan = (0., 20.)

prob = ODEProblem(diels_alder, u0, tspan, p)

sol = solve(prob, Tsit5())

plot(sol)


observed = (u,p,t) -> [100.0 * u[3]*M[3]/(u[1] * M[1] + u[2] * M[2] + u[4] * M[4]) ]

tgrid = collect(0.0:0.25:20.0)[1:end-1]

control_T = ControlParameter(tgrid, name=:temperature, controls = 20. * ones(length(tgrid)), bounds = (20.0, 100.0))

dt = 0.5

oed_layer = OEDLayer(prob, Tsit5();
            params = [1,2],
            tunable_ic = [1,2,4],
            bounds_ic = ([0.0,0.0,0.4], [10.0,10.0, 9.0]),
            controls = (control_T,),
            control_indices = [6],
            observed = observed, dt=dt)


ps, st = LuxCore.setup(Random.default_rng(), oed_layer)
lb, ub = Corleone.get_bounds(oed_layer)
pp = ComponentArray(ps)

# Bounds on the concentration of the catalyst
lb.p[6] = 0.0
ub.p[6] = 10.0

nc = vcat(0, cumsum(reduce(vcat, [length(x.t)] for x in oed_layer.layer.controls)))

constraints = let nc=nc, st = st, ax = getaxes(pp), dt = dt
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sampling = sum(ps.controls[nc[2]+1:nc[3]]) * dt
        initial_mass = ps.u0[1]*M[1] + ps.u0[2]*M[2] + ps.u0[3]*M[4]
        initial_active_ingredients = ps.u0[1]*M[1] + ps.u0[2]*M[2] / (ps.u0[1]*M[1] + ps.u0[2]*M[2] + ps.u0[3]*M[4])
        res .= [sampling, initial_mass, initial_active_ingredients]
    end
end

constraints(zeros(3), pp, st)

crit = ACriterion()
ACrit = crit(oed_layer)
ACrit(pp, nothing)

optfun = OptimizationFunction(
    ACrit, AutoForwardDiff(), cons = constraints
)

optprob = OptimizationProblem(
    optfun, collect(pp), lb = collect(lb), ub = collect(ub), lcons=[0.0, 0.1, 0.1], ucons=[6.0, 10.0, 0.7]
)

uopt = solve(optprob, Ipopt.Optimizer(),
    #tol = 1e-9,
    hessian_approximation = "limited-memory",
    max_iter = 300
)

optu = uopt + zero(pp)
constraints(zeros(3), optu, st)

optsol, _ = oed_layer(nothing, optu, st)

label_species = ["n1", "n2", "n3", "n4"]

f = Figure()
ax = CairoMakie.Axis(f[1,1], xticks = 0:5:20, title="States")
ax2 = CairoMakie.Axis(f[1,2], xticks = 0:5:20, title="Sensitivities")
ax3 = CairoMakie.Axis(f[2,1], xticks = 0:5:20, title="Temperature")
ax4 = CairoMakie.Axis(f[2,2], xticks = 0:5:20, title="Sampling")
[plot!(ax, optsol.t, sol, label=label_species[i]) for (i,sol) in enumerate(eachrow(Array(optsol))[1:4])]
axislegend(ax)
[plot!(ax2, optsol.t, sol) for sol in eachrow(reduce(hcat, (optsol[Corleone.sensitivity_variables(oed_layer)])))]
stairs!(ax3, tgrid, optu.controls[nc[1]+1:nc[2]])
stairs!(ax4, 0.0:dt:last(tspan)-dt, optu.controls[nc[2]+1:nc[3]])
f