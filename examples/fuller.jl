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


function fuller(du, u, p, t)
    du[1] = u[2]
    du[2] = 1.0 - 2.0 * p[1]
    du[3] = u[1]^2
end

tspan = (0.0, 1.0)
u0 = [1e-2, 0.0, 0.0]
p = [0.5]

prob =  ODEProblem(fuller, u0, tspan, p,
        abstol=1e-8, reltol=1e-8
)
plot(solve(prob, Tsit5()))

dt = 0.01
cgrid = collect(0.0:dt:1.0)[1:end-1]
control = ControlParameter(
    cgrid, name = :u, controls = rand(length(cgrid)), bounds = (0,1)
)

layer = SingleShootingLayer(prob, Tsit5(), controls= (1 => control,))

ps, st = LuxCore.setup(Random.default_rng(), layer)
cp = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer) .|> ComponentArray

loss = let layer = layer, st = st, ax = getaxes(cp)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        last(sols.u)[3]
    end
end

cons = let layer = layer, st = st, ax = getaxes(cp)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        return last(sols.u)[1:2]
    end
end

eq_cons(res, _x, _p) = res  .= cons(_x,_p)

optfun = OptimizationFunction(
    loss, AutoForwardDiff(), cons=eq_cons
)

optprob = OptimizationProblem(
    optfun, collect(cp), lb = collect(lb), ub = collect(ub),
    lcons = [1e-2, 0.0],
    ucons = [1e-2, 0.0]
)

uopt = solve(optprob, Ipopt.Optimizer(),
    tol = 1e-12,
    hessian_approximation = "limited-memory",
    max_iter = 250
)

optsol, _ = layer(nothing, uopt + zero(cp), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
scatterlines!(ax, optsol,vars=[:x₁, :x₂, :x₃])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2,1])
stairs!(ax1, optsol, vars=[:u₁])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
display(f)