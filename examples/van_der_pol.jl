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
# using blockSQP


function van_der_pol(du, u, p, t)
    x1, x2 = u
    du[1] = (1 - x2^2) * x1 - x2 + p[1]
    du[2] = x1
    du[3] = x1^2 + x2^2 + p[1]^2
end

tspan = (0.0, 10.)
u0 = [0.0, 1.0, 0.0]
p = [0.5]

prob =  ODEProblem(van_der_pol, u0, tspan, p,
        abstol=1e-8, reltol=1e-8
)

dt = 0.1
cgrid = collect(0.0:dt:last(tspan))[1:end-1]
control = ControlParameter(
    cgrid, name = :control, controls= 0.5 * ones(length(cgrid)), bounds = (-1.,1.)
)

layer = SingleShootingLayer(prob, Tsit5(), controls= (1 => control, ))

ps, st = LuxCore.setup(Random.default_rng(), layer)

constraint_grid = collect(0.0:0.5:last(tspan))[2:end]
constraints = Dict(
    :x₁ => (t=constraint_grid,
            bounds = (-0.25 * ones(length(constraint_grid)), 10 * ones(length(constraint_grid)))
    ),
)

sol, _ = layer(nothing, ps, st)

optprob = OptimizationProblem(
    layer, :x₃, constraints=constraints
)

uopt = solve(optprob, Ipopt.Optimizer(),
    tol = 1e-6,
    #hessian_approximation = "limited-memory",
    max_iter = 250
)

optu = uopt + zero(ComponentArray(ps))
optsol, _ = layer(nothing, uopt + zero(ComponentArray(ps)), st)

cons = zero(optprob.lcons)
optprob.f.cons(cons, uopt.u, optprob.p)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
scatterlines!(ax, optsol,vars=[:x₁, :x₂, :x₃])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2,1])
stairs!(ax1, optsol, vars=[:u₁])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
ax2 = CairoMakie.Axis(f[3,1])
scatterlines!(ax2, constraint_grid, cons, label="Constraint")
hlines!(ax2, optprob.lcons, color=:gray, linestyle=:dash, label="Lower bound")
f[3, 2] = Legend(f, ax2, "Constraints", framevisible = false)
display(f)
