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

function egerstedt(du, u, p, t)
    x,y,_ = u
    u1, u2, u3 = p
    du[1] = -x * u1 + (x + y) * u2 + (x - y) * u3;
    du[2] = (x + 2 * y) * u1 + (x - 2 * y) * u2 + (x + y) * u3;
    du[3] = x^2 + y^2
end

tspan = (0., 1.0)
u0 = [0.5, 0.5, 0.0]
p = 1/3 * ones(3)

prob =  ODEProblem(egerstedt, u0, tspan, p)

N = 20
cgrid = collect(LinRange(tspan..., N+1))[1:end-1]
c1 = ControlParameter(
    cgrid, name = :u1, controls = zeros(N) .+ 1/2, bounds = (0.,1.)
)
c2 = ControlParameter(
    cgrid, name = :u2, controls = zeros(N) .+ 1/3 , bounds = (0.,1.)
)
c3 = ControlParameter(
    cgrid, name = :u3, controls = zeros(N) .+ 1/4, bounds = (0.,1.)
)

layer = Corleone.SingleShootingLayer(prob, Tsit5(), controls=([1,2,3] .=> [c1,c2,c3]))
ps, st = LuxCore.setup(Random.default_rng(), layer)
p = ComponentArray(ps)
lb, ub = Corleone.get_bounds(layer) .|> ComponentArray

loss = let layer = layer, st = st, ax = getaxes(p)
    (p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        sols, _ = layer(nothing, ps, st)
        last(sols.u)[3]
    end
end

loss(collect(p), nothing)

sampling_cons = let layer = layer, st = st, nc=N, ax = getaxes(p)
    (res, p, ::Any) -> begin
        ps = ComponentArray(p, ax)
        res .= [ps.controls[i] + ps.controls[i+nc] + ps.controls[i+2*nc] for i = 1:nc]
    end
end

optfun = OptimizationFunction(
    loss, AutoForwardDiff(), cons=sampling_cons
)

optprob = OptimizationProblem(
    optfun, collect(p), lb = collect(lb), ub = collect(ub), lcons=ones(N), ucons=ones(N)
)
uopt = solve(optprob, Ipopt.Optimizer(),
     tol = 1e-6,
     hessian_approximation = "limited-memory",
     max_iter = 300
)

optsol, _ = layer(nothing, uopt + zero(p), st)

f = Figure()
ax = CairoMakie.Axis(f[1,1])
scatterlines!(ax, optsol, vars=[:x₁,:x₂,:x₃])
f[1, 2] = Legend(f, ax, "States", framevisible = false)
ax1 = CairoMakie.Axis(f[2,1])
stairs!(ax1, optsol, vars=[:u₁, :u₂, :u₃])
f[2, 2] = Legend(f, ax1, "Controls", framevisible = false)
f